/*
 * Decompile binaries.
*/


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import ghidra.app.decompiler.*;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;


public class Decompiler extends GhidraScript {

    private static final boolean SKIP_PARAMETERS_WITH_UNKNOWN_TYPE = false;
    private static final boolean SKIP_EXTERNAL_FUNCTIONS = false;
    private static final boolean FORMAL_SIGNATURE = false;
    private static final boolean INCLUDE_CALLING_CONVENTION = true;

    private int timeoutPerFile = -1;
    private int timeoutPerFunc = -1;

    @Override
    protected void run() throws Exception {
	// Log configuration.
        println("run: SKIP_PARAMETERS_WITH_UNKNOWN_TYPE=" + String.valueOf(SKIP_PARAMETERS_WITH_UNKNOWN_TYPE));
        println("run: SKIP_EXTERNAL_FUNCTIONS=" + String.valueOf(SKIP_EXTERNAL_FUNCTIONS));
        println("run: FORMAL_SIGNATURE=" + String.valueOf(FORMAL_SIGNATURE));
        println("run: INCLUDE_CALLING_CONVENTION=" + String.valueOf(INCLUDE_CALLING_CONVENTION));

        // Get the command line arguments.
        String[] scriptArgs = getScriptArgs();
        if (scriptArgs == null || scriptArgs.length < 3) {
            println("Error: outputDir, timeoutPerFile, timeoutPerFunc required.");
            return;
        }
        String outputDir = scriptArgs[0];
        this.timeoutPerFile = Integer.parseInt(scriptArgs[1]);
        this.timeoutPerFunc = Integer.parseInt(scriptArgs[2]);
	println("run: outputDir=" + outputDir);
	println("run: timeoutPerFile=" + String.valueOf(this.timeoutPerFile));
        println("run: timeoutPerFunc=" + String.valueOf(this.timeoutPerFunc));

        // Get the current program.
        Program program = getCurrentProgram();
        if (program == null) {
            println("Error: no program loaded.");
            return;
        }
        String programName = program.getName();
        if (programName.contains(".")) {
            programName = programName.substring(0, programName.lastIndexOf('.'));
        }
	println("run: programName=" + programName);

        // Get the output file.
        File dir = new File(outputDir);
        if (!dir.exists()) {
            println("Error: output directory does not exist.");
            return;
        }
        String outputFileName = outputDir + File.separator + programName + ".c";
	println("run: outputFileName=" + outputFileName);

        // Get the functions to process.
	FunctionIterator functions;
	if (SKIP_EXTERNAL_FUNCTIONS) {
	    functions = program.getFunctionManager().getFunctions(true);
	} else {
	    functions = program.getListing().getFunctions(true);
	}

        // Set up the DecompilerInterface.
        DecompInterface decompInterface = new DecompInterface();
        decompInterface.openProgram(program);

        // Run the main worker function.
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Callable<Void> task = () -> {
            decompileFunctions(decompInterface, functions, outputFileName);
            return null;
        };
        Future<Void> future = executor.submit(task);
        try {
            future.get(this.timeoutPerFile, TimeUnit.SECONDS);
            println("run: decompilation finished.");
        } catch (TimeoutException e) {
            println("run: decompilation timed out.");
            future.cancel(true);
        } catch (InterruptedException | ExecutionException e) {
            println("run: decompilation crashed.");
            e.printStackTrace();
        } finally {
            decompInterface.dispose();
            executor.shutdown();
        }
    }

    private void decompileFunctions(DecompInterface decompInterface, FunctionIterator functions, String outputFileName) {
        String decompiledCode;
	try (FileWriter writer = new FileWriter(outputFileName)) {
            for (Function func : functions) {
                decompiledCode = decompileFunction(decompInterface, func);
		writer.write(decompiledCode);
            }
        } catch (IOException e) {
            println("Failed to write output file: " + e.getMessage());
        }
    }

    private String decompileFunction(DecompInterface decompInterface, Function func) {

        DecompileResults results = decompInterface.decompileFunction(func, this.timeoutPerFunc, null);
        String signature = func.getPrototypeString(FORMAL_SIGNATURE, INCLUDE_CALLING_CONVENTION);

	// If decompilation was successful, we get the C code, then replace the signature
	// with the more detailed signature from Function.getPrototypeString. The default
	// signature from DecompiledFunction.getC can also be very inconsitent when it
	// chooses to include and exclude call conventions, is unclear.
	if (results.decompileCompleted()) {
            DecompiledFunction decompiledFunc = results.getDecompiledFunction();
            String decompiledCode = decompiledFunc.getC();
	    String[] decompiledCodeLines = decompiledCode.split("\n");
            decompiledCodeLines[1] = signature;
	    decompiledCode = String.join("\n", decompiledCodeLines);
            return decompiledCode;
	}

	// If decomilation timed out, was cancelled, or failed for some other reason,
	// return the function signature along with a comment documenting the failure.
        String message;
	if (results.isTimedOut()) {  // NOTE: this doesn't seem to trigger correctly.
	    println("decompileFunction: results.isTimedOut()=true");
            message = "WARNING: Decompilation incomplete due to timeoutPerFile="
		    + String.valueOf(this.timeoutPerFile) + ".";
	} else if (results.isCancelled()) {  // NOTE: this doesn't seem to trigger correctly.
	    println("decompileFunction: results.isCancelled()=true");
	    message = "WARNING: Decompilation incomplete due to timeoutPerFunc="
		    + String.valueOf(this.timeoutPerFunc) + ".";
	} else {
	    println("decompileFunction: results.getErrorMessage()=" + results.getErrorMessage().replace("\n", ""));
	    message = "WARNING: Decompilation incomplete due to ErrorMessage=`"
		    + results.getErrorMessage().replace("\n", "") + "`.";
	}

	return signature + "\n{\n\n/* " + message + " */\n}\n\n";

    }

}

