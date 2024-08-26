#!/usr/bin/env python2
# -*- coding:utf-8 -*-


# FIXME: clean up the timeout mechanism...use signals again and os.kill(os.getpid)

import os
import sys
from ghidra.app.decompiler import DecompInterface

# Trying to use a timeout object gives `TypeError: No visible constructors for class'
# which I can't figure out how to fix, so we'll use signals instead.
# from ghidra.app.util.headless import HeadlessTimedTaskMonitor
# from ghidra.util.task import TimeoutTaskMonitor

import signal
import threading
import time

# `currentProgram` or `getScriptArgs` function is contained in `__main__`
# actually you don't need to import by yourself, but it makes much "explicit"
import __main__ as ghidra_app


def handler(signum, frame):
    raise RuntimeError("TimeoutError")


def run_with_timeout(func, timeout, *args, **kwargs):
    def wrapper(result_container):
        result_container[0] = func(*args, **kwargs)
    
    result_container = [None]
    thread = threading.Thread(target=wrapper, args=(result_container,))
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False
        # raise RuntimeError("TimeoutError")
    # return result_container[0]
    return True


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

        # Enumerate all functions and decompile each function
        funcs = ghidra_app.currentProgram.getListing().getFunctions(True)
        for func in funcs:
            dec_func = self.decompile_func(func)
            if dec_func:
                yield dec_func


def run():

    # getScriptArgs gets argument for this python script using `analyzeHeadless`
    args = ghidra_app.getScriptArgs()

    cur_program_name = ghidra_app.currentProgram.getName()
    outfile = '{}.c'.format(''.join(cur_program_name.split('.')[:-1]))
    outdir = args[0] if len(args) >= 1 else ""
    output = os.path.join(outdir, outfile)
    output_timeout = output.replace(".c", ".c.timeout")

    timeout = int(args[1].strip()) if len(args) == 2 else None

    # print("outfile:" + outfile)
    # print("outdir:" + outdir)
    # print("output:" + output)
    # print("output_timout:" + output_timeout)

    decompiler = Decompiler()

    def _run():
        with open(output, "w") as fw:
            for dec_func in decompiler.decompile():
                fw.write(dec_func)

    # try:
    #     run_with_timeout(_run, timeout)
    # except RuntimeError:
    #     os.rename(output, output_timeout)
    #     print('[*] success (timeout). save to -> {}'.format(output_timeout))

    if run_with_timeout(_run, timeout):
        print('[*] success. save to -> {}'.format(output))
    else:
        print('[*] success (timeout). save to -> {}'.format(output_timeout))
        decompiler._decompiler.stopProcess()  # :) Fuck that was painful.
        os.rename(output, output_timeout)


# Starts execution here
if __name__ == '__main__':
    run()
