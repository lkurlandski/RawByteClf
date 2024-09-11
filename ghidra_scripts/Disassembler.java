/**
 * Lift binaries to disassembly.
*/

import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.address.Address;
import ghidra.program.model.mem.MemoryAccessException;

public class Disassembler extends Lifter {

    @Override
    protected String processFunction(Function func) throws Exception {
        return disassembleFunction(func);
    }

    @Override
    protected String getFileExtension() {
        return ".asm";
    }

    private String disassembleFunction(Function func) throws MemoryAccessException { // FIXME: investigate getListing...

        Address funcAddr = func.getEntryPoint();
        Iterator<Instruction> instructions = currentProgram.getListing().getInstructions(funcAddr, true);

        String signature = func.getPrototypeString(FORMAL_SIGNATURE, INCLUDE_CALLING_CONVENTION);
        String disassembledCode = "\n" + signature + "\n";

        while (instructions.hasNext()) {
            Instruction inst = instructions.next();
            if (currentProgram.getFunctionManager().getFunctionContaining(inst.getAddress()) != func) {
                break;
            }
            disassembledCode = disassembledCode + formatInstruction(inst);
	}

	return disassembledCode;
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
