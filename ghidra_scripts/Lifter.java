/**
 * Lift binaries to intermediate representation.
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

import ghidra.app.util.headless.HeadlessScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Program;

public abstract class Lifter extends HeadlessScript {

    protected static final boolean SKIP_EXTERNAL_FUNCTIONS = false;
    protected static final boolean FORMAL_SIGNATURE = false;
    protected static final boolean INCLUDE_CALLING_CONVENTION = true;
    protected static final boolean REPLACE_SIGNATURE = true;
    protected static final boolean REQUIRE_HEADLESS_ANALYSIS_COMPLETE = false;

    protected String outputDir = "./";
    protected int timeoutPerFile = -1;
    protected int timeoutPerFunc = -1;

    /**
     * Entry point.
    */
    @Override
    protected void run() throws Exception {
        println("run: SKIP_EXTERNAL_FUNCTIONS=" + SKIP_EXTERNAL_FUNCTIONS);
        println("run: FORMAL_SIGNATURE=" + FORMAL_SIGNATURE);
        println("run: INCLUDE_CALLING_CONVENTION=" + INCLUDE_CALLING_CONVENTION);
        println("run: REPLACE_SIGNATURE=" + REPLACE_SIGNATURE);
        println("run: REQUIRE_HEADLESS_ANALYSIS_COMPLETE=" + REQUIRE_HEADLESS_ANALYSIS_COMPLETE);
        println("run: analysisTimeoutOccurred()=" + analysisTimeoutOccurred());

	if (REQUIRE_HEADLESS_ANALYSIS_COMPLETE && !analysisTimeoutOccurred()) {
            println("run: skipped.");
	    return;
        }
        
        String[] scriptArgs = getAndValidateScriptArgs();
        this.outputDir = scriptArgs[0];
        this.timeoutPerFile = Integer.parseInt(scriptArgs[1]);
        this.timeoutPerFunc = Integer.parseInt(scriptArgs[2]);
        println("run: outputDir=" + this.outputDir);
        println("run: timeoutPerFile=" + String.valueOf(this.timeoutPerFile));
        println("run: timeoutPerFunc=" + String.valueOf(this.timeoutPerFunc));

        Program program = getCurrentProgram();
        if (program == null) {
            println("run: no program.");
            return;
        }
 
        String programName = getProgramName(program);
        String outputFileName = getOutputFileName(this.outputDir, programName);        
        FunctionIterator functions = getFunctions(program);
        runMainWorker(functions, outputFileName);
    }

    /**
     * Process a single function and write the output to a file.
    */
    protected abstract String processFunction(Function func) throws Exception;
    
    /**
     * Get the output file's extension, e.g., ".EXTENSION"
    */
    protected abstract String getFileExtension();

    /**
     * Get the command line arguments passed to the instance.
    */
    private String[] getAndValidateScriptArgs() {
        String[] scriptArgs = getScriptArgs();
        if (scriptArgs == null || scriptArgs.length < 3) {
            throw new IllegalArgumentException("Error: outputDir, timeoutPerFile, timeoutPerFunc required.");
        }
        return scriptArgs;
    }

    /**
     * Get the name of this file, excluding the extension, i.e., the SHA-256.
    */
    private String getProgramName(Program program) {
        String programName = program.getName();
        if (programName.contains(".")) {
            programName = programName.substring(0, programName.lastIndexOf('.'));
        }
        println("run: programName=" + programName);
        return programName;
    }

    /**
     * Get the name of the output file for this file.
    */
    private String getOutputFileName(String outputDir, String programName) {
        File dir = new File(outputDir);
        if (!dir.exists()) {
            throw new IllegalArgumentException("Error: output directory does not exist.");
        }
        String outputFileName = outputDir + File.separator + programName + getFileExtension();
        println("run: outputFileName=" + outputFileName);
        return outputFileName;
    }

    /**
     * Get an iterator of functions to process.
    */
    private FunctionIterator getFunctions(Program program) {
        if (SKIP_EXTERNAL_FUNCTIONS) {
            return program.getFunctionManager().getFunctions(true);
        } else {
            return program.getListing().getFunctions(true);
        }
    }

    /**
     * Wraps processFunctions in a timeout construct.
    */
    private void runMainWorker(FunctionIterator functions, String outputFileName) throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Callable<Void> task = () -> {
            processFunctions(functions, outputFileName);
            return null;
        };
        Future<Void> future = executor.submit(task);
        try {
            future.get(this.timeoutPerFile, TimeUnit.SECONDS);
            println("run: finished.");
        } catch (TimeoutException e) {
            println("run: timed out.");
            future.cancel(true);
        } catch (InterruptedException | ExecutionException e) {
            println("run: crashed.");
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Process every function and write the output of processing to a file.
    */
    private void processFunctions(FunctionIterator functions, String outputFileName) throws Exception {
	String processedFunc;
        try (FileWriter writer = new FileWriter(outputFileName)) {
            for (Function func : functions) {
                processedFunc = processFunction(func);
		writer.write(processedFunc);
            }
        } catch (IOException e) {
            throw e;
        }
    }
}
