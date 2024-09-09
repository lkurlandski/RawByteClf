import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Program;
import ghidra.program.model.address.Address;
import ghidra.program.model.mem.MemoryAccessException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

public class Disassembler extends GhidraScript {

    private static final boolean SKIP_PARAMETERS_WITH_UNKNOWN_TYPE = false;
    private static final boolean SKIP_EXTERNAL_FUNCTIONS = false;

    @Override
    public void run() throws Exception {

	// Get the command line arguments.
        String[] scriptArgs = getScriptArgs();
        if (scriptArgs == null || scriptArgs.length < 1) {
            println("Error: input directory required.");
            return;
        }

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

        // Get the output file.
        String outputDir = scriptArgs[0];
        File dir = new File(outputDir);
        if (!dir.exists()) {
            println("Error: output directory does not exist.");
            return;
        }
        String outputFileName = outputDir + File.separator + programName + ".asm";

	// Get the functions.
	FunctionIterator functions;
	if (SKIP_EXTERNAL_FUNCTIONS) {
            functions = program.getFunctionManager().getFunctions(true);
	} else {
	    functions = program.getListing().getFunctions(true);
	}

	// Disassemble the functions.
        try (FileWriter writer = new FileWriter(outputFileName)) {
            for (Function func : functions) {
                disassembleFunction(func, writer);
            }
        } catch (IOException e) {
            println("Error: could not write to file: " + e.getMessage());
	    return;
        }
    }

    private void disassembleFunction(Function func, FileWriter writer) throws IOException, MemoryAccessException {
        Address funcAddr = func.getEntryPoint();
        Iterator<Instruction> instructions = currentProgram.getListing().getInstructions(funcAddr, true);

        // Write function signature
        writer.write("\n" + getFunctionSignature(func) + "\n");

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

    private String getFunctionSignature(Function func) {
        String callingConv = func.getCallingConventionName();
        if (callingConv.equals("unknown")) {
	    callingConv = func.DEFAULT_CALLING_CONVENTION_STRING;
        }

        StringBuilder paramBuilder = new StringBuilder();
        for (var param : func.getParameters()) {

            if (SKIP_PARAMETERS_WITH_UNKNOWN_TYPE) {
                if (param.getDataType().getName().equals("undefined")) {
		    continue;
		}
	    }

            String paramStr = param.toString().replace("[", "").replace("]", "").split("@")[0];
            if (paramBuilder.length() > 0) {
                paramBuilder.append(", ");
            }
            paramBuilder.append(paramStr);
        }

        return String.format("%s %s(%s)", callingConv, func.getName(), paramBuilder.toString());
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

