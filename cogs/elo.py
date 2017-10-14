import discord
from discord.ext import commands
import aiohttp
import json
import pandas as pd
import datetime
import gc

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
            self.match_history = pd.read_json(self.config['match_history_path'])
        except:
            # Create a new match history
            self.match_history = pd.DataFrame(columns=['timestamp', 'playerID', 'elo', 'new_elo', 'team', 'status'])

        try:
            self.user_status = pd.read_json(self.config['user_status_path'])
        except:
            # Create new user status
            self.user_status = pd.DataFrame(columns=['name', 'elo', 'wins', 'losses', 'matches_played', 'rank', 'color'])
            self.user_status.index.name = 'playerID'

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
        for time, match in self.match_history.groupby('timestamp', as_index=False):
            match = match.copy()
            # Replace the elo rating of each player with what it should be, from the user_status
            match['elo'] = match['playerID'].apply(lambda p: self.get_elo(user_status, p))
        
            # Update the user status for each player
            user_status = await self.update_players(ctx, match, user_status)

            # Grab the new elo
            match = match.merge(user_status.reset_index()[['playerID', 'elo']].rename(columns=dict(elo='new_elo', on='playerID')))

            # Add the match to the new match history
            match_history = match_history.append(match, ignore_index=True)

        # Finally, update the Elo object's match history and user status
        self.match_history = match_history
        self.user_status = user_status

        print("Recalculated elo ratings.")
        print("New match history:")
        print(match_history)
        print("New ratings:")
        print(user_status)

    async def update_players(self, ctx, match_df, user_status):
        team_elo = match_df.groupby('team')[['elo']].sum()

        print(team_elo)
        team_elo['status'] = match_df.groupby('team').head(1).set_index('team')['status']
        print(team_elo)

        user_status = user_status.copy()

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
            await ctx.message.channel.send('Unknown team status! Try one of '+
                               (', ').join(self.config['status_values'].keys()) + '!')
            return None

        # If score limit must be met exactly...
        if self.config['require_score_limit'] and team_elo['actual'].sum() != self.config['score_limit']:
            print(team_elo['actual'].sum())
            await ctx.message.channel.send('Not enough/too many teams are winning/losing!')
            return None

        # Limit total score
        if team_elo['actual'].sum() > self.config['score_limit']:
            await ctx.message.channel.send('Maximum score exceeded! Make sure the teams are not all winning!')
            return None

        k_factor = self.config['k_factor']
        team_elo['elo_delta'] = k_factor * (team_elo['actual'] - team_elo['expected'])

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
        

    @commands.command()
    async def match(self, ctx, *, args: str):
        '''Record a match into the system.

        format: match TEAM1 TEAM2 [at YYYY-mm-dd HH-mm-ss]

        where TEAM# is in the format @mention1 [@mention2 ...] {win|loss|draw}

        example: match @mention1 @mention2 win ! @mention3 @mention4 loss at 2017-01-01 23:01:01
        This represents a 2v2 game, where mention1 and mention2 defeated
        mention3 and mention4.

        There must be at least two teams. The team listing must end
        with a status, for example win or loss.

        This requires that the caller have permissions to manage matches.
        '''

        time_now = datetime.datetime.utcnow()
        timestamp = time_now
        match_data = []

        split_time = args.split(sep=' at ')
        if len(split_time) > 1:

            # Try various ways of formatting the time, and infer missing information.
            # If it doesn't work, then we've gotta complain instead of silently failing!
            try:
                # Full date and full time
                timestamp = datetime.datetime.strptime(split_time[1], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
            try:
                # Full date, time without seconds
                timestamp = datetime.datetime.strptime(split_time[1], '%Y-%m-%d %H:%M')
            except ValueError:
                pass
            try:
                # Full date only
                timestamp = datetime.datetime.strptime(split_time[1], '%Y-%m-%d')
            except ValueError:
                pass
            # The same without the year attached
            try:
                # Full date and full time
                timestamp = datetime.datetime.strptime(split_time[1], '%m-%d %H:%M:%S')
                timestamp = timestamp.replace(year=datetime.date.today().year)
            except ValueError:
                pass
            try:
                # Full date, time without seconds
                timestamp = datetime.datetime.strptime(split_time[1], '%m-%d %H:%M')
                timestamp = timestamp.replace(year=datetime.date.today().year)
            except ValueError:
                pass
            try:
                # Full date only
                timestamp = datetime.datetime.strptime(split_time[1], '%m-%d')
                timestamp = timestamp.replace(year=datetime.date.today().year)
            except ValueError:
                pass

        # So if all those methods failed, we complain
        if len(split_time) > 1 and timestamp == time_now:
            # Complain here!
            print('failed to parse timestamp')
            await ctx.message.channel.send('Couldn\'t parse timestamp! Make sure you follow the format!\n'
                         '(the timestamp should be formatted YYYY-mm-dd HH-mm-ss with '
                         '24 hour time.)')
            return

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
            user_id = member_str.strip('<@>')
            print(user_id)
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
                    await ctx.message.channel.send('The same user was repeated multiple times!')
                    return
            except:
                # So if this isn't a user id, it must be the team status.

                # If we just finished a team, this is an extraneous argument!
                if done_with_team:
                    await ctx.message.channel.send('Extraneous arguments detected while parsing teams!\n'
                                       'Make sure you use valid @mentions for all players '
                                       'and only specify `win` or `loss` after the list of '
                                       'players!')
                    return

                print('Status found: %s' % member_str)
                team_status = member_str
                done_with_team = True
                teams[team_name] = (team, team_status)

        # If we haven't completed all the teams (all teams must terminate with a team status)
        # then we've gotta complain!
        if not done_with_team:
            await ctx.message.channel.send('Team not completed! Make sure to terminate the teams with a team status!\n'
                         '(try putting win or loss after the list of team members)')
            return
            
        for team in teams.keys():
            members, status = teams[team]
            for member in members:
                # Get player's elo
                tm_elo = self.get_elo(self.user_status, member)

                match_data.append(dict(timestamp=timestamp, playerID=member, elo=tm_elo, team=team, status=status))
                
        
        # Create the df
        match_df = pd.DataFrame(match_data, columns=self.match_history.columns.drop('new_elo'))
        print(match_df)

        # Make sure there are actually at least 2 teams.
        if match_df['team'].nunique() < 2:
            await ctx.message.channel.send('Need at least 2 teams for a match!')
            return

        if match_df['team'].nunique() > self.config['max_teams']:
            await ctx.message.channel.send('Too many teams in this match!')
            return

        # update elo rating of player and wins/losses,
        # by calling another function
        new_user_status = await self.update_players(ctx, match_df, self.user_status)
        if new_user_status is not None:
            self.user_status = new_user_status
        else:
            # If we couldn't update the scores, fail!
            return

        # Update names of people mentioned here...
        for member in ctx.message.mentions:
            self.user_status.loc[member.id, 'name'] = member.name

        match_df = match_df.merge(self.user_status.reset_index()[['playerID', 'elo']].rename(columns=dict(elo='new_elo', on='playerID')))

        # Add the new df to the match history
        self.match_history = self.match_history.append(match_df, ignore_index=True)
        print("current match history:")
        print(self.match_history)
        print("current player ratings:")
        print(self.user_status)
        # Sometimes there are circular references within dataframes? so we have to
        # invoke the gc
        gc.collect()
        await ctx.message.channel.send('Added match!')
        await self.show_match(ctx, timestamp)

    async def show_match(self, ctx, timestamp):

        # First try to find the match
        match = self.match_history.set_index('timestamp')
        match = match.loc[pd.Timestamp(timestamp)]
        if len(match) == 0:
            # We couldn't find the match!
            return False

        # Now that we have the match, we pretty-print

        # We can set the title to something like 1v1 Match
        title = 'v'.join(match.groupby('team')['playerID'].count().astype(str).tolist())
        title += ' match'
        embed = discord.Embed(title=title, type='rich', timestamp=timestamp)

        for team, team_members in match.groupby('team'):
            field_name = 'Team %s (%s)' % (team, team_members['status'].iloc[0])
            field_value = ''
            for i, t in team_members.iterrows():
                field_value += '*%s* (%d -> %d)\n' % (self.user_status.loc[t['playerID'], 'name'], round(t['elo']), round(t['new_elo']))
            embed.add_field(name=field_name, value=field_value)
        return await ctx.message.channel.send(embed=embed)

    async def get_player_card(self, ctx, user_id):
        '''Get an Embed describing the player's stats.

        ctx is a context from which we can grab the server and
        the avatar.

        user_id is of the player whose stats are to be shown.
        '''

        if user_id in self.user_status.index:
            uinfo = self.user_status.loc[user_id]
        else:
            return None
        try:
            avatar = ctx.message.server.get_member(user_id).avatar_url
        except:
            avatar = None

        title = '%s (%s, %d)' % (uinfo['name'], uinfo['rank'], int(uinfo['elo']))
        embed = discord.Embed(type='rich',
                              color=int('0x' + uinfo['color'], base=16))
        if avatar:
            embed.set_author(name=title, icon_url=avatar)
        else:
            embed.set_author(name=title)
        embed.add_field(name='Wins', value=int(uinfo['wins']))
        embed.add_field(name='Losses', value=int(uinfo['losses']))
        embed.add_field(name='Total', value=int(uinfo['matches_played']))
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
            await ctx.message.channel.send('Couldn\'t find any players!')
            return

        page_size = self.config['max_player_cards']
        # If we find only one page of players, just output them.
        if len(player_cards) <= page_size:
            for card in player_cards:
                await ctx.message.channel.send(embed=card)
        # If we find more than one page, show the page number as well
        else:
            page_count = (len(player_cards) + page_size - 1) / page_size
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
            await ctx.message.channel.send('The number of top players to show must be an integer!')
            return

        # Make sure the number is non-negative
        if n < 0:
            await ctx.message.channel.send('Cannot display a negative number of top players!')
            return

        # Make sure the number doesn't exceed the configurable limit
        if n > self.config['max_top']:
            await ctx.message.channel.send('Maximum players to display in top rankings is %d!'\
                    % self.config['max_top'])
            return

        topn = self.user_status.sort_values('elo', ascending=False).head(n)
        title = 'Top %d Players' % n
        desc = ''

        for i, (uid, uinfo) in enumerate(topn.iterrows()):
            desc += '%d. %s (%s, %d)\n' % (i+1, uinfo['name'], uinfo['rank'], round(uinfo['elo']))
        embed = discord.Embed(title=title, type='rich', description=desc)

        return await ctx.message.channel.send(embed=embed)


