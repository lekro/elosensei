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
            self.user_status = pd.DataFrame(columns=['elo', 'wins', 'losses'])
            self.user_status.index.name = 'playerID'

    def get_elo(self, user_status, player):
        if player in user_status.index:
            return user_status.loc[player, 'elo']
        else:
            user_status.loc[player] = (self.config['default_elo'], 0, 0)
            return self.config['default_elo']

    async def recalculate_elo(self):
        
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
            user_status = await self.update_players(match, user_status)

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

    async def update_players(self, match_df, user_status):
        team_elo = match_df.groupby('team')[['elo']].sum()

        team_elo['status'] = match_df.set_index('team')['status']

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
            await self.bot.say('Unknown team status! Try one of '+
                               (', ').join(self.config['status_values'].keys()) + '!')
            return None

        # If score limit must be met exactly...
        if self.config['require_score_limit'] and team_elo['actual'].sum() != self.config['score_limit']:
            await self.bot.say('Not enough/too many teams are winning/losing!')
            return None

        # Limit total score
        if team_elo['actual'].sum() > self.config['score_limit']:
            await self.bot.say('Maximum score exceeded! Make sure the teams are not all winning!')
            return None

        k_factor = self.config['k_factor']
        team_elo['elo_delta'] = k_factor * (team_elo['actual'] - team_elo['expected'])

        print("processing one match:")
        print(team_elo)

        for index, row in match_df.iterrows():
            player = row['playerID']
            user_status.loc[player, 'elo'] += team_elo.loc[row.team, 'elo_delta']
            actual_score = team_elo.loc[row.team, 'actual']
            if actual_score == 1:
                user_status.loc[player, 'wins'] += 1
            elif actual_score == 0:
                user_status.loc[player, 'losses'] += 1

        return user_status


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
        

    @commands.command(pass_context=True)
    async def match(self, ctx, *, args: str):
        '''Record a match into the system.

        format: match @mention1 [@others_on_team_1 ...] {win|loss|draw} ! @mention2 [@others ...] {win|loss|draw} [! other teams in the same format ...] [at YYYY-mm-dd HH-mm-ss]

        example: match @mention1 @mention2 win ! @mention3 @mention4 loss at 2017-01-01 23:01:01
        This represents a 2v2 game, where mention1 and mention2 defeated
        mention3 and mention4.

        There must be at least two teams. The team listing must end
        with a status, for example win or loss.

        This requires that the caller have permissions to manage matches.
        '''

        time_now = datetime.datetime.now()
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
            await self.bot.say('Couldn\'t parse timestamp! Make sure you follow the format!\n'
                         '(the timestamp should be formatted YYYY-mm-dd HH-mm-ss with '
                         '24 hour time.)')
            return

        teams_str = split_time[0]
        
        team_name = 0

        for team_str in teams_str.split(sep='!'):
            team_name += 1
            team = []
            done_with_team = False
            for member_str in team_str.split():
                # If we have already iterated through, that means there are extraneous
                # arguments! Notify the user that they will be ignored...
                if done_with_team:
                    await self.bot.say('Extraneous arguments detected while parsing teams!\n'
                                       'Make sure you use valid @mentions for all players '
                                       'and only specify `win` or `loss` after the list of '
                                       'players!')
                    return
                # First make sure this is actually a valid user...
                user_id = member_str.strip('<@>')
                print(user_id)
                try:
                    # The user id should be an integer as a string...
                    # We could optionally query discord, but that takes an annoying
                    # amount of time.
                    int(user_id)
                    team.append(user_id)
                except:
                    # So if this isn't a user id, it must be the team status.
                    team_status = member_str
                    done_with_team = True
            for team_member in team:
                # Get player's elo
                tm_elo = self.get_elo(self.user_status, team_member)

                match_data.append(dict(timestamp=timestamp, playerID=team_member, elo=tm_elo, team=team_name, status=team_status))
                
        
        # Create the df
        match_df = pd.DataFrame(match_data, columns=self.match_history.columns.drop('new_elo'))
        print(match_df)

        # Make sure there are actually at least 2 teams.
        if match_df['team'].nunique() < 2:
            await self.bot.say('Need at least 2 teams for a match!')
            return

        if match_df['team'].nunique() > self.config['max_teams']:
            await self.bot.say('Too many teams in this match!')
            return

        # update elo rating of player and wins/losses,
        # by calling another function
        new_user_status = await self.update_players(match_df, self.user_status)
        if new_user_status is not None:
            self.user_status = new_user_status
        else:
            # If we couldn't update the scores, fail!
            return

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
        await self.bot.say('Added match!')

    @commands.command(pass_context=True)
    async def recalculate(self, ctx):
        '''Recalculate elo ratings from scratch.'''
        await self.recalculate_elo()
        await self.bot.say('Recalculated elo ratings!')



