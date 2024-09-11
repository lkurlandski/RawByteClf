/**
 * Lift binaries to disassembly.
*/

import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
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

    private String disassembleFunction(Function func) throws MemoryAccessException, IllegalArgumentException {

        Address funcAddr = func.getEntryPoint();
        InstructionIterator instructions = currentProgram.getListing().getInstructions(funcAddr, true);
        FunctionManager funcManager = currentProgram.getFunctionManager();

        String signature = func.getPrototypeString(FORMAL_SIGNATURE, INCLUDE_CALLING_CONVENTION);
        String disassembledCode = "\n" + signature + "\n";

        while (instructions.hasNext()) {
            Instruction inst = instructions.next();    
            if (funcManager.getFunctionContaining(inst.getAddress()) != func) {
                break;
            }
            disassembledCode = disassembledCode + formatInstruction(inst);
	}

	return disassembledCode;
    }

    private String formatInstruction(Instruction inst) throws MemoryAccessException, IllegalArgumentException {
	Address virtAddr = inst.getAddress();
	Address physAddr = virtAddr.getPhysicalAddress();

	String sectStr = inst.getAddressString(true, true).split(":")[0];
	sectStr = sectStr.substring(0, Math.min(sectStr.length(), 8));
        String virtAddrStr = virtAddr.toString(true, true).split(":")[1];
        String physAddrStr = physAddr.toString(true, true).split(":")[1];
        String byteStr = formatByteStr(inst.getBytes());
        String instStr = inst.toString();

        if (sectStr.length() > 8) {  // 8 characters for the section name.
	    throw new IllegalArgumentException("StringTooLong: sectStr=" + sectStr);
	}
        if (virtAddrStr.length() > 16) {  // 16 characters for the physical address (maximum for 64-bit address space).
	    throw new IllegalArgumentException("StringTooLong: virtAddrStr=" + virtAddrStr);
	}
        if (physAddrStr.length() > 16) {  // 16 characters for the virtual address (maximum for 64-bit address space).
	    throw new IllegalArgumentException("StringTooLong: physAddrStr=" + physAddrStr);
	}
        if (byteStr.length() > 48) {  // 48 characters for the raw-bytes (maximum 16 byte instructions).
	    throw new IllegalArgumentException("StringTooLong: byteStr=" + byteStr);
	}

	return String.format("%-8s\t%-16s\t%-16s\t%-48s\t%s\n", sectStr,  physAddrStr, virtAddrStr, byteStr, instStr);
    }

    private String formatByteStr(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x ", b & 0xff));
        }
        return sb.toString().trim();
    }
}
