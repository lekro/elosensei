import discord
from discord.ext import commands
import aiohttp
import json
import pandas as pd
import datetime
import gc
import asyncio


class EloError(Exception):
    '''An error for Elo rating meant to be presented to the user nicely'''

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

async def on_command_error(ctx, error):
    '''Global error handler which gives EloError back to the user'''

    original = error.original

    if isinstance(original, EloError):
        await ctx.message.channel.send(original)
    else:
        raise original

class Elo:
    '''
    Elo rating commands from Elo-sensei
    '''

    def __init__(self, bot, config):
        self.bot = bot
        self.config = config['elo']

        # We also need to load the dataframes for both the match history
        # and current users status, given paths in the config
        try:
            self.match_history = pd.read_pickle(self.config['match_history_path'])
        except OSError:
            # Create a new match history
            self.match_history = pd.DataFrame(columns=['timestamp', 'eventID', 'playerID', 'elo', 'new_elo', 'team', 'status', 'value', 'comment'])

        try:
            self.user_status = pd.read_pickle(self.config['user_status_path'])
        except OSError:
            # Create new user status
            self.user_status = pd.DataFrame(columns=['name', 'elo', 'wins', 'losses', 'matches_played', 'rank', 'color'])
            self.user_status.index.name = 'playerID'

            # Set categorical dtype for rank
            self.user_status['rank'] = self.user_status['rank'].astype('category')

            # Get all the possible ranks and add them to the categorical type
            all_ranks = [rank['name'] for rank in self.config['ranks']]
            self.user_status['rank'] = self.user_status['rank'].cat.add_categories(all_ranks)

        # Create locks to prevent race conditions during asynchronous operation
        self.user_status_lock = asyncio.Lock()
        self.match_history_lock = asyncio.Lock()

        # Handle command errors...
        bot.on_command_error = on_command_error

    def get_elo(self, user_status, player):
        if player in user_status.index:
            return user_status.loc[player, 'elo']
        else:
            user_status.loc[player] = dict(name=None, elo=self.config['default_elo'], wins=0, losses=0, matches_played=0, rank=None, color=None)
            return self.config['default_elo']

    async def recalculate_elo(self, ctx):
        
        # Reinstantiate user status
        user_status = pd.DataFrame(columns=self.user_status.columns)
        user_status.index.name = 'playerID'
        # Reinstantiate match history
        match_history = pd.DataFrame(columns=self.match_history.columns)

        # For each match...
        await self.match_history_lock.acquire()
        for time, match in self.match_history.groupby('timestamp', as_index=False):
            match = match.copy()
            # Replace the elo rating of each player with what it should be, from the user_status
            match['elo'] = match['playerID'].apply(lambda p: self.get_elo(user_status, p))
        
            # Update the user status for each player
            user_status = await self.update_players(ctx, match, user_status)

            # Grab the new elo
            match = match.merge(user_status.reset_index()[['playerID', 'elo']].rename(columns=dict(elo='new_elo')), on='playerID')

            # Add the match to the new match history
            match_history = match_history.append(match, ignore_index=True)

        # Finally, update the Elo object's match history and user status
        await self.user_status_lock.acquire()
        self.match_history = match_history
        self.user_status = user_status
        self.match_history_lock.release()
        self.user_status_lock.release()

        print("Recalculated elo ratings.")
        print("New match history:")
        print(match_history)
        print("New ratings:")
        print(user_status)

    async def process_single_player_events(self, match_df, user_status, lock=None):

        event = match_df.iloc[0]
        if lock is not None:
            await lock.acquire()
        elo = self.get_elo(user_status, event['playerID'])
        if event['status'] == 'delta':
            user_status.loc[event['playerID'], 'elo'] += event['value']
        elif event['status'] == 'set':
            user_status.loc[event['playerID'], 'elo'] = event['value']

        await self.update_rank(user_status, event['playerID'])
        if lock is not None:
            lock.release()

        return user_status



    async def update_players(self, ctx, match_df, user_status, lock=None):

        # If this isn't a match, process it as a single player event (e.g. score adjustment)
        if len(match_df) == 1:
            return await self.process_single_player_events(match_df, user_status, lock=None)

        # Otherwise, continue to process it as a normal match
        team_elo = match_df.groupby('team')[['elo']].sum()

        print(team_elo)
        # Bring in the first values of status and value for each team
        team_elo[['status', 'value']] = match_df.groupby('team').head(1).set_index('team')[['status', 'value']]
        print(team_elo)

        if lock is not None:
            await lock.acquire()
        user_status = user_status.copy()
        if lock is not None:
            lock.release()

        # Take mean of every team but this one
        for index, row in team_elo.iterrows():
            team_elo.loc[index, 'other_elo'] = team_elo.drop(index)['elo'].mean()
        
        # Expected score for teams
        # This uses the logistic curve and formulas from Wikipedia
        team_elo['expected'] = 1./(1.+10.**((team_elo['other_elo'] - team_elo['elo'])/400))

        # Actual team scores
        team_elo['actual'] = team_elo['status'].apply(self.get_status_value)

        # If allowing only defined status values, we might have NaN values in there...
        # Fail if that happens..
        if team_elo['actual'].isnull().any():
            raise EloError('Unknown team status! Try one of '+
                               (', ').join(self.config['status_values'].keys()) + '!')

        # If score limit must be met exactly...
        if self.config['require_score_limit'] and team_elo['actual'].sum() != self.config['score_limit']:
            print(team_elo['actual'].sum())
            raise EloError('Not enough/too many teams are winning/losing!')

        # Limit total score
        if team_elo['actual'].sum() > self.config['score_limit']:
            raise EloError('Maximum score exceeded! Make sure the teams are not all winning!')

        team_elo['elo_delta'] = team_elo['value'] * (team_elo['actual'] - team_elo['expected'])

        print("processing one match:")
        print(team_elo)

        for index, row in match_df.iterrows():
            player = row['playerID']
            user_status.loc[player, 'elo'] += team_elo.loc[row.team, 'elo_delta']
            actual_score = team_elo.loc[row.team, 'actual']
            user_status.loc[player, 'matches_played'] += 1
            if actual_score == 1:
                user_status.loc[player, 'wins'] += 1
            elif actual_score == 0:
                user_status.loc[player, 'losses'] += 1
            await self.update_rank(user_status, row['playerID'])

        return user_status

    async def update_rank(self, user_status, uid):
        max_rank = None
        for rank in self.config['ranks']:
            if max_rank is None:
                if user_status.loc[uid, 'elo'] > rank['cutoff'] or rank['default']:
                    max_rank = rank
            else:
                if user_status.loc[uid, 'elo'] > rank['cutoff'] and rank['cutoff'] > max_rank['cutoff']:
                    max_rank = rank
        user_status.loc[uid, 'rank'] = max_rank['name']
        user_status.loc[uid, 'color'] = max_rank['color']

    def get_status_value(self, status):
        try:
            return self.config['status_values'][status]
        except:
            if self.config['allow_only_defined_status_values']:
                # This will become NaN in the dataframe
                # and we can catch it later
                return None
            else:
                return self.config['default_status_value']
        

    @commands.command(name='match')
    async def match(self, ctx, *, args: str):
        '''Record a match into the system.

        format: match TEAM1 TEAM2 [at [YYYY-]mm-dd HH-mm] [K=value] ["title and comments"]

        where TEAM# is in the format @mention1 [@mention2 ...] {win|loss|draw}

        example: match @mention1 @mention2 win @mention3 @mention4 loss at 2017-01-01 23:01 K=55 "foo"
        This represents a 2v2 game, where mention1 and mention2 defeated
        mention3 and mention4 at 23:01 UTC on January 1, 2017, with a
        K factor of 55, and an occasion of foo.

        There must be at least two teams. The team listing must end
        with a status, for example win or loss.

        This requires that the caller have permissions to manage matches.
        '''

        time_now = datetime.datetime.utcnow()
        timestamp = time_now
        match_data = []
        
        # Find comment, if any.
        args = args.split('"')
        if len(args) > 1:
            if len(args) == 3:
                # The user has a comment to make!
                comment = args[1]
            else:
                # Mismatched quotes or multiple comments?
                raise EloError('Comments must be enclosed in "double quotes!"')
        else:
            # No comment to make
            comment = None
        # Get first arg, ignore comments
        args = args[0]

        # Find custom K factor, if any.
        args = args.split(' K=')
        if len(args) > 1:
            # The user input a custom K factor
            try:
                k_factor = float(args[1])
            except ValueError:
                raise EloError('K factor must be a number!')
        else:
            # The user wants the default K factor
            k_factor = self.config['k_factor']
        # Get the first argument, ignore K factor part
        args = args[0]

        split_time = args.split(sep=' at ')
        if len(split_time) > 1:

            '''Only accept a timestamp in the format [YYYY-]mm-dd hh:mm
               It's just simpler so why not...'''
            # If it doesn't work, then we've gotta complain instead of silently failing!
            try:
                # Full date and full time
                timestamp = datetime.datetime.strptime(split_time[1], '%Y-%m-%d %H:%M')
            except ValueError:
                pass
            # The same without the year attached
            try:
                # Date and time but year ommited: fill in with current year
                timestamp = datetime.datetime.strptime(split_time[1], '%m-%d %H:%M')
                timestamp = timestamp.replace(year=datetime.date.today().year)
            except ValueError:
                pass
            
        # So if all those methods failed, we complain
        if len(split_time) > 1 and timestamp == time_now:
            # Complain here!
            print('failed to parse timestamp')
            raise EloError('Couldn\'t parse timestamp! Make sure you follow the format!\n'
                         '(the timestamp should be formatted [YYYY]-mm-dd hh:mm with '
                         '24 hour time.)')
        #If timestamp is valid, but in the future... complain!
        elif timestamp > time_now:
            print('Timestamp was invalid')
            raise EloError('I may be an amazing bot, but I can\'t record matches '
                           'that are in the future!')

        teams_str = split_time[0]
        
        team_name = 1
        users_seen = []

        teams = {}
        team = []
        done_with_team = False
        for member_str in teams_str.split():
            # If we have already iterated through, that means there are extraneous
            # arguments! Notify the user that they will be ignored...
            # First make sure this is actually a valid user...
            user_id = member_str.strip('<@!>')
            print("Checking for user id: {}".format(user_id))
            try:
                # The user id should be an integer as a string...
                # We could optionally query discord, but that takes an annoying
                # amount of time.
                user_id = int(user_id)
                print('User id found: %s' % user_id)

                # Since this is a user id, start a new team if necessary:
                if done_with_team:
                    print('Done with team, adding another team...')
                    team = []
                    team_name += 1
                    done_with_team = False
                team.append(user_id)
                if user_id not in users_seen:
                    users_seen.append(user_id)
                else:
                    # We are getting a duplicate user.
                    raise EloError('The same user was repeated multiple times!')
            except ValueError:
                # So if this isn't a user id, it must be the team status.

                # If we just finished a team, this is an extraneous argument!
                if done_with_team:
                    raise EloError('Extraneous arguments detected while parsing teams!\n'
                                       'Make sure you use valid @mentions for all players '
                                       'and only specify `win` or `loss` after the list of '
                                       'players!')

                print('Status found: %s' % member_str)
                team_status = member_str
                done_with_team = True
                teams[team_name] = (team, team_status)

        # If we haven't completed all the teams (all teams must terminate with a team status)
        # then we've gotta complain!
        if not done_with_team:
            raise EloError('Team not completed! Make sure to terminate the teams with a team status!\n'
                         '(try putting win or loss after the list of team members)')
            
        await self.user_status_lock.acquire()
        for team in teams.keys():
            members, status = teams[team]
            for member in members:
                # Get player's elo
                tm_elo = self.get_elo(self.user_status, member)

                match_data.append(dict(timestamp=timestamp, playerID=member, elo=tm_elo, team=team, status=status))
        self.user_status_lock.release()
                
        
        # Create the df
        await self.match_history_lock.acquire()
        match_df = pd.DataFrame(match_data, columns=self.match_history.columns.drop('new_elo'))
        match_df['eventID'] = self.match_history['timestamp'].nunique() + 1
        match_df['value'] = k_factor
        match_df['comment'] = comment
        self.match_history_lock.release()
        print(match_df)

        # Make sure there are actually at least 2 teams.
        if match_df['team'].nunique() < 2:
            raise EloError('Need at least 2 teams for a match!')

        if match_df['team'].nunique() > self.config['max_teams']:
            raise EloError('Too many teams in this match!')

        # update elo rating of player and wins/losses,
        # by calling another function
        new_user_status = await self.update_players(ctx, match_df, self.user_status, lock=self.user_status_lock)
        if new_user_status is not None:
            await self.user_status_lock.acquire()
            self.user_status = new_user_status
            self.user_status_lock.release()
        else:
            # If we couldn't update the scores, fail!
            return

        # Update names of people mentioned here...
        await self.user_status_lock.acquire()
        
        match_players = match_df['playerID'].tolist()
        for user_id in match_players:
            try:
                member = ctx.guild.get_member(user_id)
            except ValueError:
                pass
            else:
                if member.nick != None:
                    self.user_status.loc[member.id, 'name'] = member.nick
                else:
                    self.user_status.loc[member.id, 'name'] = member.name

        match_df = match_df.merge(self.user_status.reset_index()[['playerID', 'elo']].rename(columns=dict(elo='new_elo', on='playerID')))
        self.user_status_lock.release()

        # Add the new df to the match history
        await self.match_history_lock.acquire()
        self.match_history = self.match_history.append(match_df, ignore_index=True)
        self.match_history_lock.release()
        print("current match history:")
        print(self.match_history)
        print("current player ratings:")
        print(self.user_status)
        # Sometimes there are circular references within dataframes? so we have to
        # invoke the gc
        gc.collect()
        await ctx.message.channel.send(embed=await self.get_event_embed(ctx, timestamp))

    @commands.command()
    async def show(self, ctx, *, arg):
        '''Display information for a match or event given a date or event ID.

        show [match-id-or-time] [page]

        Display event with ID #14: show 14
        Display all events on 2017-01-01: show 2017-01-01
        Display the second page of events on 2017-01-01: show 2017-01-01 2
        '''

        args = arg.split()
        if len(args) > 1:
            try:
                page = int(args[1])-1
                print('page requested: %d' % page)
            except ValueError:
                raise EloError("Page number must be an integer!")
        else:
            page = 0
        arg = args[0]

        try:
            eventID = int(arg)
        except ValueError:
            # Try to parse it as a date
            try:
                timestamp = datetime.datetime.strptime(arg, '%Y-%m-%d')
            except ValueError:
                raise EloError("Couldn't parse argument as event ID or date!")
            else:
                await self.match_history_lock.acquire()
                mask = (timestamp <= self.match_history['timestamp']) & \
                        (timestamp + datetime.timedelta(days=1) > self.match_history['timestamp'])
                timestamps = self.match_history.drop_duplicates(subset='timestamp').loc[mask, 'timestamp'].dt.to_pydatetime()
                del mask
                self.match_history_lock.release()
        else:
            await self.match_history_lock.acquire()
            mask = self.match_history['eventID'] == eventID
            timestamps = self.match_history.drop_duplicates(subset='timestamp').loc[mask, 'timestamp'].dt.to_pydatetime()
            self.match_history_lock.release()

        if len(timestamps) < 1:
            raise EloError("No events found!")

        print(timestamps)

        event_cards = [await self.get_event_embed(ctx, ts) for ts in timestamps]

        page_size = self.config['max_match_cards']

        # If we find only one page of players, just output them.
        if len(event_cards) <= page_size:
            for card in event_cards:
                await ctx.message.channel.send(embed=card)
        # If we find more than one page, show the page number as well
        else:
            page_count = (len(event_cards) + page_size - 1) / page_size
            if not (0 <= page < page_count):
                raise EloError("Page index out of range!")
            # Iterate through the player cards only in the page we want...
            for i, card in enumerate(event_cards[page*page_size:(page+1)*page_size]):
                if i==0:
                    page_string = 'Showing page %d of %d of event cards.' % (page+1, page_count)
                else:
                    page_string = ''
                await ctx.message.channel.send(page_string, embed=card)


    async def get_event_embed(self, ctx, timestamp):

        # First try to find the match
        await self.match_history_lock.acquire()
        match = self.match_history.set_index('timestamp').loc[pd.Timestamp(timestamp)].copy()
        self.match_history_lock.release()
        if len(match) == 0:
            # We couldn't find the match!
            return False

        # Now that we have the match, we pretty-print

        # We can set the title to something like 1v1 Match
        desc_text = 'v'.join(match.groupby('team')['playerID'].count().astype(str).tolist())
        desc_text += ' match'
        if match['comment'].iloc[0] is not None:
            title = match['comment'].iloc[0]
        else:
            title = desc_text

        desc_text += ' (K=%d)' % match['value'].iloc[0]
        embed = discord.Embed(title=title, description=desc_text, type='rich', timestamp=timestamp)

        for team, team_members in match.groupby('team'):
            field_name = 'Team %s (%s)' % (team, team_members['status'].iloc[0])
            field_value = ''
            for i, t in team_members.iterrows():
                field_value += '*%s* (%d -> %d)\n' % (self.user_status.loc[t['playerID'], 'name'], round(t['elo']), round(t['new_elo']))
            embed.add_field(name=field_name, value=field_value)

        # Show eventID
        embed.set_footer(text='#%d' % match['eventID'].iloc[0])
        return embed

    async def show_events(self, ctx, timestamps):

        pass

    async def get_player_card(self, ctx, user_id):
        '''Get an Embed describing the player's stats.

        ctx is a context from which we can grab the server and
        the avatar.

        user_id is of the player whose stats are to be shown.
        '''

        await self.user_status_lock.acquire()
        if user_id in self.user_status.index:
            uinfo = self.user_status.loc[user_id]
            self.user_status_lock.release()
        else:
            self.user_status_lock.release()
            return None
        try:
            avatar = ctx.guild.get_member(user_id).avatar_url
        except:
            avatar = None

        title = '%s (%s, %d)' % (uinfo['name'], uinfo['rank'], int(uinfo['elo']))

        # Construct description field
        description = "Wins: %d / Losses: %d / Total: %d\n" % (uinfo['wins'], uinfo['losses'], uinfo['matches_played'])
        description += "Player ID: %s\n" % user_id

        # Get all matches played
        await self.match_history_lock.acquire()
        ids_played = self.match_history.query('playerID == %s' % user_id)['eventID'].tolist()
        ids_played = [str(i) for i in ids_played]
        self.match_history_lock.release()
        description += "Events: %s\n" % (', '.join(ids_played))
        embed = discord.Embed(type='rich', description=description, color=int('0x' + uinfo['color'], base=16))
        if avatar:
            embed.set_author(name=title, icon_url=avatar)
        else:
            embed.set_author(name=title)
        return embed


    @commands.command()
    async def recalculate(self, ctx):
        '''Recalculate elo ratings from scratch.'''
        await self.recalculate_elo(ctx)
        await ctx.message.channel.send('Recalculated elo ratings!')

    @commands.command()
    async def player(self, ctx, *, name=None):
        '''Show a player's Elo profile.

        Players can be searched by the beginning of their name, or by mentioning
        them. If no search query is present, the caller's (your) player card
        will be shown, if any.

        For example, `elo! player lekro` will display the profile of all
        players whose names start with 'lekro'. 

        You can also @mention user(s).
        '''

        # For now, we'll only search the database of known users. 
        # But we can also check the server itself.
        # TODO check the server and get ids of users with this name as well
        player_cards = []
        if name is not None:
            # Get page number to display. This will be the last part of the name,
            # if any.

            try:
                page = int(name.split()[-1])-1
                # Remove the page number from the query
                name = name[:name.rfind(' ')]
            except ValueError:
                # Assume we want to see the first page
                page = 0

            for i, (uid, uinfo) in enumerate(self.user_status.iterrows()):
                if str(uinfo['name']).lower().startswith(name.lower()):
                    player_cards.append(await self.get_player_card(ctx, uid))
            # Process mentions 
            if len(ctx.message.mentions) > 0:
                for mention in ctx.message.mentions:
                    card = await self.get_player_card(ctx, mention.id)
                    if card is not None:
                        player_cards.append(card)
        else:
            # Process self
            card = await self.get_player_card(ctx, ctx.message.author.id)
            if card is not None:
                player_cards.append(card)

        # If we found no players, tell the caller that!
        if len(player_cards) == 0:
            raise EloError('Couldn\'t find any players!')

        page_size = self.config['max_player_cards']
        # If we find only one page of players, just output them.
        if len(player_cards) <= page_size:
            for card in player_cards:
                await ctx.message.channel.send(embed=card)
        # If we find more than one page, show the page number as well
        else:
            page_count = (len(player_cards) + page_size - 1) / page_size
            if not (0 <= page < page_count):
                raise EloError("Page index out of range!")
            # Iterate through the player cards only in the page we want...
            for i, card in enumerate(player_cards[page*page_size:(page+1)*page_size]):
                if i==0:
                    page_string = 'Showing page %d of %d of player cards.' % (page+1, page_count)
                else:
                    page_string = ''
                await ctx.message.channel.send(page_string, embed=card)


    @commands.command()
    async def top(self, ctx, *, n=10):
        '''Show the top n players.'''

        # Make sure the input is an integer
        try:
            n = int(n)
        except ValueError:
            raise EloError('The number of top players to show must be an integer!')

        # Make sure the number is non-negative
        if n < 0:
            raise EloError('Cannot display a negative number of top players!')

        # Make sure the number doesn't exceed the configurable limit
        if n > self.config['max_top']:
            raise EloError('Maximum players to display in top rankings is %d!'\
                    % self.config['max_top'])

        await self.user_status_lock.acquire()
        print(self.user_status)
        topn = self.user_status.sort_values('elo', ascending=False).head(n)
        self.user_status_lock.release()
        title = 'Top %d Players' % n
        desc = ''
        print(topn)
        for i, (uid, uinfo) in enumerate(topn.iterrows()):
            print(i)
            desc += '%d. %s (%s, %d)\n' % (i+1, uinfo['name'], uinfo['rank'], round(uinfo['elo']))
        print(title)
        print(desc)
        embed = discord.Embed(title=title, type='rich', description=desc)

        return await ctx.message.channel.send(embed=embed)

    @commands.command()
    async def top(self, ctx, *, n=10):
        '''Show the top n players.'''

        try:
            await self.top_command(ctx, n)
        except EloError as e:
            await ctx.message.channel.send(e)
