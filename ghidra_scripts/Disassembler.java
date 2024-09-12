/**
 * Lift binaries to disassembly.
*/

import java.io.FileWriter;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Iterator;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.address.Address;
import ghidra.program.model.mem.MemoryAccessException;

public class Disassembler extends Lifter {

    public int bitSize;
    public int bitMult;
    public String unknownAddressStr;

    @Override
    protected void run() throws Exception {

        this.bitSize = currentProgram.getLanguage().getDefaultSpace().getSize();
        if (this.bitSize == 16) {
            this.bitMult = 1;
        } else if (this.bitSize == 32) {
            this.bitMult = 2;
        } else if (this.bitSize == 64) {
            this.bitMult = 4;
        } else {
            throw new IllegalArgumentException("Invalid word size: size=" + String.valueOf(this.bitSize));
        }
	this.unknownAddressStr = "?".repeat(this.bitMult * 4);

        super.run();
    }

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
            String sectionName = getSectionName(inst);
            String physAddr = getPhysicalAddress(inst);
            String virtAddr = getVirtualAddress(inst);
            String bytes = getBytes(inst);
            String instruction = getInstruction(inst);
            String line = formatInstruction(sectionName, physAddr, virtAddr, bytes, instruction);
            disassembledCode = disassembledCode + line;
        }

        return disassembledCode;
    }

    private String getSectionName(Instruction inst) {
        return inst.getAddressString(true, true).split(":")[0];
    }

    private String getPhysicalAddress(Instruction inst) {
        Address virtAddr = inst.getAddress();
	long physAddr;
	try {
	    physAddr = virtualAddressToPhysicalAddress(virtAddr);
	} catch (ArithmeticException e) {
            return this.unknownAddressStr;
        }

	if (physAddr > 4294967295L) {
	    return this.unknownAddressStr;
	}
        return String.format("%08x", physAddr);
    }


    /**
    * Convert a virtual address to a physical one.
    */
    private long virtualAddressToPhysicalAddress(Address addr) throws ArithmeticException {
        Memory memory = currentProgram.getMemory();
        MemoryBlock block = memory.getBlock(addr); 
	long sectionOffset = addr.getOffsetAsBigInteger()
		           .subtract(block.getStart().getOffsetAsBigInteger())
			   .longValueExact();
	long physAddr = block.getSourceInfos().get(0).getFileBytesOffset()
		      + sectionOffset;
	if (physAddr < 0) {
	    throw new ArithmeticException();
	}
	return physAddr;
    }

    private String getVirtualAddress(Instruction inst) {
        return inst.getAddressString(true, true).split(":")[1];
    }

    private String getBytes(Instruction inst) throws MemoryAccessException {
        byte[] bytes = inst.getBytes();
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x ", b & 0xff));
        }
        return sb.toString().trim();
    }

    private String getInstruction(Instruction inst) {
        return inst.toString();
    }

    private String formatInstruction(String sectionName, String physAddr, String virtAddr, String bytes, String instruction) throws IllegalArgumentException {

        if (sectionName.length() > 8) {  // 8 characters for the section name.
            sectionName = sectionName.substring(0, 8);
        }
        if (physAddr.length() > this.bitMult * 4) {  // Maximum for `size`-bit address space.
            throw new IllegalArgumentException("StringTooLong: physAddr=" + physAddr);
        }
        if (virtAddr.length() > this.bitMult * 4) {  // Maximum for `size`-bit address space.
            throw new IllegalArgumentException("StringTooLong: virtAddr=" + virtAddr);
        }
        if (bytes.length() > 48) {  // 48 characters for up to 15 bytes per instruction.
            throw new IllegalArgumentException("StringTooLong: bytes=" + bytes);
        }

        String format = "%-8s\t%-"
		      + String.valueOf(this.bitMult * 4)
		      + "s\t%-"
		      + String.valueOf(this.bitMult * 4)
		      + "s\t%-48s\t%s\n";
        return String.format(format, sectionName,  physAddr, virtAddr, bytes, instruction);
    }
}
