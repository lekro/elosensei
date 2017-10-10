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
            self.match_history = pd.DataFrame(columns=['timestamp', 'playerID', 'elo', 'team', 'status'])

        try:
            self.user_status = pd.read_json(self.config['user_status_path'])
        except:
            # Create new user status
            self.user_status = pd.DataFrame(columns=['elo', 'wins', 'losses'])

    def get_elo(self, user_status, player):
        try:
            return user_status.loc[player, 'elo']
        except:
            user_status.loc[player] = (self.config['default_elo'], 0, 0)
            return self.config['default_elo']

    async def recalculate_elo(self):
        
        # Reinstantiate user status
        user_status = pd.DataFrame(columns=self.user_status.columns)
        # Reinstantiate match history
        match_history = pd.DataFrame(columns=self.match_history.columns)

        # For each match...
        for time, match in self.match_history.groupby('timestamp'):
            match = match.copy()
            # Replace the elo rating of each player with what it should be, from the user_status
            match['elo'] = match['playerID'].applymap(lambda p: self.get_elo(user_status, p))
        
            # Update the user status for each player
            await self.update_players(match, user_status)

        # Finally, update the Elo object's match history and user status
        self.match_history = match_history
        self.user_status = user_status
            

    async def update_players(self, match_df, user_status):
        team_elo = match_df.groupby('team')[['elo', 'status']].sum(numeric_only=True)

        score_diff = team_elo.iloc[0, 'elo'] - team_elo.iloc[1, 'elo']

        # Take mean of every team but this one
        for index, row in team_elo.iterrows():
            team_elo.loc[index, 'other_elo'] = team_elo.drop(index)['elo'].mean()
        
        # Expected score for teams
        team_elo['expected'] = 1./(1.+10.**(team_elo['other_elo'] - team_elo['elo']))

        # Actual team scores
        team_elo['actual'] = team_elo['status'].applymap(self.get_status_value)
        k_factor = self.config['k_factor']
        team_elo['elo_delta'] = k_factor * (team_elo['actual'] - team_elo['expected'])
        
        for index, row in match_df.iterrows():
            user_status.loc[player, 'elo'] += team_elo.loc[row.team, 'elo_delta']
            actual_score = team_elo.loc[row.team, 'actual']
            if actual_score == 1:
                user_status.loc[player, 'wins'] += 1
            elif actual_score == 0:
                user_status.loc[player, 'losses'] += 1


    def get_status_value(self, status):
        try:
            return self.config['status_values'][status]
        except:
            return self.config['default_status_value']
        

    @bot.command(pass_context=True)
    async def match(self, ctx, *, args: str):
        '''Record a match into the system.

        format: match @mention1 @mention2 win ! @mention3 @mention4 loss
        This represents a 2v2 game, where mention1 and mention2 defeated
        mention3 and mention4.

        This requires that the caller have permissions to manage matches.
        '''

        time_now = datetime.datetime.now()
        match_data = []

        team_name = 0

        for team_str in args.split(sep='!'):
            team_name += 1
            team = []
            for member_str in team_str.rstrip().split(sep=' '):
                team.append(member_str)
            team_status = team.pop()
            for team_member in team:
                # Get player's elo
                tm_elo = self.get_elo(self.user_status, team_member)

                match_data.append([time_now, team_member, tm_elo, team_name, team_status])
                
        
        # Create the df
        match_df = pd.DataFrame(match_data, columns=self.match_history.columns)

        # update elo rating of player and wins/losses,
        # by calling another function
        await self.update_players(match_df, self.user_status)

        # Add the new df to the match history
        self.match_history = self.match_history.append(match_df)
        # Sometimes there are circular references within dataframes? so we have to
        # invoke the gc
        gc.collect()
        await self.bot.say('Added match!')
        await self.bot.say(str(match_df))

    @bot.command(pass_context=True)
    async def recalculate(self, ctx):
        '''Recalculate elo ratings from scratch.'''
        await recalculate_elo(self)
        await self.bot.say('Recalculated elo ratings!')



