import discord
from discord.ext import commands
import asyncio
import json
import datetime
import logging

# Import 'cogs'
import cogs.elo

# Make sure discord stuff is put into another log
# This code is taken (almost) directly from the discord.py docs
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:'
                                       '%(name)s: %(message)s'))
logger.addHandler(handler)

# Load the config file.
with open('config.json', 'r') as jsonconf:
    config = json.load(jsonconf)

# Grab description and prefix from config
description = config['general']['description']
prefix = config['general']['prefix']

# If we wish to require a space after the prefix, add it now
if config['general']['space_after_prefix']:
    prefix += ' '

# Instantiate our bot
bot = commands.Bot(command_prefix=prefix, description=description)

# Store startup time
bot.startup_time = datetime.datetime.now()

@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('Invite: https://discordapp.com/oauth2/authorize?client_id={}&scope=bot'.format(bot.user.id))
    print('------')
    await bot.change_presence(game=discord.Game(name=config['general']['playing']))


@bot.command()
async def uptime(ctx):
    '''Check bot uptime.'''
    await ctx.message.channel.send(datetime.datetime.now() - bot.startup_time)


@bot.command
async def about(ctx):
    '''Get information about the bot.'''
    
    bot_info = '''Elo-sensei is a Discord bot aiming to automate
    the computation of Elo ratings for players in a Discord guild.
    It has been designed for use with ranked MissileWars on CubeKrowd,
    but aims to act as a general-purpose Elo rating bot.

    A manual is available at https://github.com/lekro/elosensei/wiki
    The source code is available at https://github.com/lekro/elosensei

    LEGAL:
    Copyright 2017, lekro and Elo-sensei contributors. Licensed under
    the GNU GPLv3.

    Elo-sensei collects and stores user information, including but
    not limited to usernames, nicknames, roles and other Discord
    and game related information. However, no messages are stored 
    except information derived from messages prefixed with the
    bot prefix as provided in the configuration files. Voice-related
    information is not collected at all.
    '''

    # Enclose in backticks and send
    ctx.message.channel.send('```' + bot_info + '```')

def load_cogs(bot, config):
    if config['elo']['enable']:
        bot.add_cog(cogs.elo.Elo(bot, config))



load_cogs(bot, config)
bot.run(config['general']['token'])

