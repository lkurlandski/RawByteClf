/*
* Disassemble binaries.
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
import java.util.Iterator;

import ghidra.app.util.headless.HeadlessScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Program;
import ghidra.program.model.address.Address;
import ghidra.program.model.mem.MemoryAccessException;


public class Disassembler extends HeadlessScript {

    private static final boolean SKIP_EXTERNAL_FUNCTIONS = false;
    private static final boolean FORMAL_SIGNATURE = false;
    private static final boolean INCLUDE_CALLING_CONVENTION = true;
    private static final boolean REPLACE_SIGNATURE = true;
    private static final boolean REQUIRE_HEADLESS_ANALYSIS_COMPLETE = false;

    private int timeoutPerFile = -1;
    private int timeoutPerFunc = -1;

    @Override
    protected void run() throws Exception {
        // Log configuration.
        println("run: SKIP_EXTERNAL_FUNCTIONS=" + String.valueOf(SKIP_EXTERNAL_FUNCTIONS));
        println("run: FORMAL_SIGNATURE=" + String.valueOf(FORMAL_SIGNATURE));
        println("run: INCLUDE_CALLING_CONVENTION=" + String.valueOf(INCLUDE_CALLING_CONVENTION));
        println("run: REPLACE_SIGNATURE=" + String.valueOf(REPLACE_SIGNATURE));
        println("run: REQUIRE_HEADLESS_ANALYSIS_COMPLETE=" + String.valueOf(REQUIRE_HEADLESS_ANALYSIS_COMPLETE));

        println("run: analysisTimeoutOccurred()=" + String.valueOf(analysisTimeoutOccurred()));
        if (REQUIRE_HEADLESS_ANALYSIS_COMPLETE && !analysisTimeoutOccurred()) {
            println("run: skipped.");
        }

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
        String outputFileName = outputDir + File.separator + programName + ".asm";
        println("run: outputFileName=" + outputFileName);

        // Get the functions to process.
        FunctionIterator functions;
        if (SKIP_EXTERNAL_FUNCTIONS) {
            functions = program.getFunctionManager().getFunctions(true);
        } else {
            functions = program.getListing().getFunctions(true);
        }

        // Run the main worker function.
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Callable<Void> task = () -> {
            disassembleFunctions(functions, outputFileName);
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

    private void disassembleFunctions(FunctionIterator functions, String outputFileName) throws Exception {

        try (FileWriter writer = new FileWriter(outputFileName)) {
            for (Function func : functions) {
                disassembleFunction(func, writer);
            }
        } catch (IOException e) {
            throw e;
        }

    }

    private void disassembleFunction(Function func, FileWriter writer) throws IOException, MemoryAccessException {
        Address funcAddr = func.getEntryPoint();
        Iterator<Instruction> instructions = currentProgram.getListing().getInstructions(funcAddr, true);

        // Write function signature
	String signature = func.getPrototypeString(FORMAL_SIGNATURE, INCLUDE_CALLING_CONVENTION);
        writer.write("\n" + signature + "\n");

        // Iterate through instructions
        while (instructions.hasNext()) {
            Instruction inst = instructions.next();
            if (currentProgram.getFunctionManager().getFunctionContaining(inst.getAddress()) != func) {
                break;
            }
            // Format and write each instruction
            writer.write(formatInstruction(inst));
        }
    }

    private String formatInstruction(Instruction inst) throws MemoryAccessException {
        String addr = inst.getAddressString(true, true);
        byte[] bytes = inst.getBytes();
        String bytecode = formatBytes(bytes);
        String instStr = inst.toString();
        return String.format(" %-15s %-30s %s\n", addr, bytecode, instStr);
    }

    private String formatBytes(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x ", b & 0xff));
        }
        return sb.toString().trim();
    }

}
