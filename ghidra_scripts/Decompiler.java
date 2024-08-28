// Decompiler.java
import ghidra.app.decompiler.*;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Decompiler extends GhidraScript {

    @Override
    protected void run() throws Exception {

        // Get the command line arguments.
        String[] scriptArgs = getScriptArgs();
        if (scriptArgs == null || scriptArgs.length < 1) {
            println("Error: input directory and timeout required.");
            return;
        }

        int timeout = Integer.parseInt(scriptArgs[1]);

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

        // Determine the file to write to.
        String outputDir = scriptArgs[0];
        File dir = new File(outputDir);
        if (!dir.exists()) {
            println("Error: output directory does not exist.");
            return;
        }
        String outputFileName = outputDir + File.separator + programName + ".c";

        // Set up the DecompilerInterface.
        DecompInterface decompInterface = new DecompInterface();
        decompInterface.openProgram(program);

        // Write each function to the output file.
        FunctionIterator functions = program.getFunctionManager().getFunctions(true);
        try (FileWriter writer = new FileWriter(outputFileName)) {
            for (Function func : functions) {
                decompileFunction(decompInterface, func, timeout, writer);
            }
        } catch (IOException e) {
            println("Failed to write output file: " + e.getMessage());
        }

        // Close the interface.
        decompInterface.dispose();
    }

    private void decompileFunction(DecompInterface decompInterface, Function func, int timeout, FileWriter writer) {
        try {
            DecompileResults results = decompInterface.decompileFunction(func, timeout, monitor);
            DecompiledFunction decompiledFunc = results.getDecompiledFunction();

            if (decompiledFunc == null) {
                // Get the function signature
                String functionSignature = func.getPrototypeString(true, false);
        
                // Write the function signature along with the timeout warning
                writer.write(functionSignature + "\n{\n\n");
                writer.write("/* WARNING: Could not decompile function within timeout */\n");
                writer.write("}\n\n");

                println("Failed to decompile function: " + func.getName() + " within timeout.");
                return;
            }

            // Write the decompiled function code
            String decompiledCode = decompiledFunc.getC();
            writer.write(decompiledCode);

        } catch (IOException e) {
            println("Failed to write decompiled function: " + e.getMessage());
        }
    }

    /*
    private void decompileFunction(DecompInterface decompInterface, Function func, int timeout, FileWriter writer) {
        try {

            DecompileResults results = decompInterface.decompileFunction(func, timeout, monitor);
            DecompiledFunction decompiledFunc = results.getDecompiledFunction();

            if (decompiledFunc == null) {
                println("Failed to decompile function: " + func.getName());
                return;
            }

            String decompiledCode = decompiledFunc.getC();
            writer.write(decompiledCode);

        } catch (IOException e) {
            println("Failed to write decompiled function: " + e.getMessage());
        }
    }
    */
}

