import discord
from discord.ext import commands
import asyncio

class Eggs:
    '''Miscellaneous commands'''

    def __init__(self, bot, config):

        self.config = config['eggs']


    @commands.command()
    async def egg(self, ctx, name=None):

        if name is None:
            name = self.config['default_egg']

        if name in self.config['eggs']:
            path = self.config['eggs'][name]
        else:
            await ctx.message.channel.send('Unknown egg with name `{}`!'.format(name))

        try:
            with open(path, 'rb') as f:
                my_file = discord.File(f)
                await ctx.message.channel.send(file=my_file)
        except OSError:
            await ctx.message.channel.send('The file for the egg named `{}` is missing!'.format(name))


