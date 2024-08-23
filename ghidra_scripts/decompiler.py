#!/usr/bin/env python2
# -*- coding:utf-8 -*-

import os
from ghidra.app.decompiler import DecompInterface

# `currentProgram` or `getScriptArgs` function is contained in `__main__`
# actually you don't need to import by yourself, but it makes much "explicit"
import __main__ as ghidra_app


class Decompiler:
    '''decompile binary into pseudo c using Ghidra API.
    Usage:
        >>> decompiler = Decompiler()
        >>> pseudo_c = decompiler.decompile()
        >>> # then write to file
    '''

    def __init__(self, program=None, timeout=None):
        '''init Decompiler class.
        Args:
            program (ghidra.program.model.listing.Program): target program to decompile, 
                default is `currentProgram`.
            timeout (ghidra.util.task.TaskMonitor): timeout for DecompInterface::decompileFunction
        '''

        # Initialize decompiler with current program
        self._decompiler = DecompInterface()
        self._decompiler.openProgram(program or ghidra_app.currentProgram)

        self._timeout = timeout
    
    def decompile_func(self, func):
        '''decompile one function.
        Args:
            func (ghidra.program.model.listing.Function): function to be decompiled
        Returns:
            string: decompiled pseudo C code
        '''

        # Decompile
        dec_status = self._decompiler.decompileFunction(func, 0, self._timeout)
        # Check if it's successfully decompiled
        if dec_status and dec_status.decompileCompleted():
            # Get pseudo C code
            dec_ret = dec_status.getDecompiledFunction()
            if dec_ret:
                return dec_ret.getC()

    def decompile(self):
        '''decompile all function recognized by Ghidra.
        Returns:
            string: decompiled all function as pseudo C
        '''

        # All decompiled result will be joined
        pseudo_c = ''

        # Enumerate all functions and decompile each function
        funcs = ghidra_app.currentProgram.getListing().getFunctions(True)
        for func in funcs:
            dec_func = self.decompile_func(func)
            if dec_func:
                pseudo_c += dec_func

        return pseudo_c


def run():

    # getScriptArgs gets argument for this python script using `analyzeHeadless`
    args = ghidra_app.getScriptArgs()

    cur_program_name = ghidra_app.currentProgram.getName()
    outfile = '{}.asm'.format(''.join(cur_program_name.split('.')[:-1]))
    outdir = args[0] if len(args) == 1 else ""
    output = os.path.join(outdir, outfile)

    # Do decompilation process
    decompiler = Decompiler()
    pseudo_c = decompiler.decompile()

    # Save to output file
    with open(output, 'w') as fw:
        fw.write(pseudo_c)
        print('[*] success. save to -> {}'.format(output))


# Starts execution here
if __name__ == '__main__':
    run()
