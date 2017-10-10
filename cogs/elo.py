import discord
from discord.ext import commands
import aiohttp
import json
import pandas

class Elo:
    '''
    Elo rating commands from Elo-sensei
    '''

    def __init__(self, bot, config):
        self.bot = bot
        self.config = config

