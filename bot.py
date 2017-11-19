import discord
from discord.ext import commands
import asyncio
import json
import datetime
import logging
from contextlib import suppress

# Import 'cogs'
import cogs.elo
import cogs.eggs

# Load the config file.
with open('config.json', 'r') as jsonconf:
    config = json.load(jsonconf)

# Make sure discord stuff is put into another log
# This code is taken (almost) directly from the discord.py docs
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:'
                                       '%(name)s: %(message)s'))
logger.addHandler(handler)

# Have a logger for Elo...
elo_logger = logging.getLogger('elo')
elo_logger.setLevel(config['general']['log_level'])
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('(%(name)s) %(asctime)s [%(levelname)s] %(message)s'))
elo_logger.addHandler(sh)

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
    elo_logger.info('Logged in as {} ({})'.format(bot.user.name, bot.user.id))
    elo_logger.info('Invite: https://discordapp.com/oauth2/authorize?client_id={}&scope=bot'.format(bot.user.id))
    await bot.change_presence(game=discord.Game(name=config['general']['playing']))


@bot.event
async def on_command(ctx):
    # Log commands sent...
    elo_logger.info('{} ({}) ran command: {}'.format(ctx.message.author.name,
        ctx.message.author.id, ctx.message.content))


@bot.command()
async def uptime(ctx):
    '''Check bot uptime.'''
    await ctx.message.channel.send(datetime.datetime.now() - bot.startup_time)


@bot.command()
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
    await ctx.message.channel.send('```' + bot_info + '```')

def load_cogs(bot, config):
    if config['elo']['enable']:
        bot.add_cog(cogs.elo.Elo(bot, config))
    if config['eggs']['enable']:
        bot.add_cog(cogs.eggs.Eggs(bot, config))


load_cogs(bot, config)

# Don't use discord.py's event loop abstraction since
# we need to cancel periodic save tasks...

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(bot.start(config['general']['token']))
except KeyboardInterrupt:

    # Now we can execute shutdown tasks for all of our cogs...
    for cog in bot.cogs:
        cog = bot.get_cog(cog)
        if hasattr(cog, 'do_shutdown_tasks'):
            loop.run_until_complete(cog.do_shutdown_tasks())

    # Logout
    loop.run_until_complete(bot.logout())

    # Here we can cancel the periodic save task...
    pending_tasks = asyncio.Task.all_tasks()
    for task in pending_tasks:
        task.cancel()
        # We now flagged the task for cancellation but we have to allow it to run.
        with suppress(asyncio.CancelledError):
            loop.run_until_complete(task)
finally:
    loop.close()

