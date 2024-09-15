/**
 * Find regions of the PE file that are executable.
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
import java.util.ArrayList;
import java.util.List;

import ghidra.app.util.headless.HeadlessScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.lang.Language;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Listing;

public class ExtractExecutableRegions extends HeadlessScript {

    protected static final boolean DETAILED_SECTIONS = false;
    protected static final boolean REQUIRE_HEADLESS_ANALYSIS_COMPLETE = false;

    protected int timeoutPerFile = -1;
    protected String programName;

    @Override
    public void run() throws Exception {
        println("run: DETAILED_SECTIONS=" + DETAILED_SECTIONS);
        println("run: REQUIRE_HEADLESS_ANALYSIS_COMPLETE=" + REQUIRE_HEADLESS_ANALYSIS_COMPLETE);
        println("run: analysisTimeoutOccurred()=" + analysisTimeoutOccurred());

        if (REQUIRE_HEADLESS_ANALYSIS_COMPLETE && !analysisTimeoutOccurred()) {
            println("run: skipped.");
            return;
        }

        String[] scriptArgs = getAndValidateScriptArgs();
        String outputFileName = scriptArgs[0];
        this.timeoutPerFile = Integer.parseInt(scriptArgs[1]);
        this.programName = getProgramName();
        println("run: outputFileName=" + outputFileName);
        println("run: timeoutPerFile=" + String.valueOf(this.timeoutPerFile));
        println("run: programName=" + this.programName);

        runMainWorker(outputFileName);
    }

    private void MainWorker(String outputFileName) throws Exception {

        Memory memory = currentProgram.getMemory();
        Listing listing = currentProgram.getListing();

        List<Bounds> execBounds = new ArrayList<>();
        List<Bounds> codeBounds = new ArrayList<>();
        List<Bounds> dataBounds = new ArrayList<>();
        List<Bounds> paddBounds = new ArrayList<>();

        for (MemoryBlock block : memory.getBlocks()) {
            if (block.isExecute()) {
                Address lower = block.getStart();
                Address upper = block.getEnd();
                Regions regions = getRegions(listing, lower, upper);
                execBounds.addAll(regions.execBounds);
                codeBounds.addAll(regions.codeBounds);
                dataBounds.addAll(regions.dataBounds);
                paddBounds.addAll(regions.paddBounds);
            }
        }

        Regions allRegions = new Regions(execBounds, codeBounds, dataBounds, paddBounds);
        String allRegionsStr = regionsToJson(allRegions);
        String outputStr = "{"
                         + "\"sha\": " + "\"" + this.programName + "\""
                         + ", "
                         + "\"regions\": " + allRegionsStr 
                         + "}"
                         + "\n";
        try (FileWriter writer = new FileWriter(outputFileName, true)) {
            writer.write(outputStr);
        } catch (IOException e) {
            throw e;
        }
    }

    /**
    * Wraps the main function in a timeout construct.
    */
    private void runMainWorker(String outputFileName) throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Callable<Void> task = () -> {
            MainWorker(outputFileName);
            return null;
        };
        Future<Void> future = executor.submit(task);
        try {
            future.get(this.timeoutPerFile, TimeUnit.SECONDS);
            println("run: finished (success) <" + this.programName + ">");
        } catch (TimeoutException e) {
            println("run: finished (timeout) <" + this.programName + ">");
            future.cancel(true);
        } catch (InterruptedException | ExecutionException e) {
            println("run: finished (crash) <" + this.programName + ">");
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }
 
    /**
     * Get the command line arguments passed to the instance.
    */
    private String[] getAndValidateScriptArgs() {
        String[] scriptArgs = getScriptArgs();
        if (scriptArgs == null || scriptArgs.length < 2) {
            println("getAndValidateScriptArgs: scriptArgs=" + String.join(", ", scriptArgs));
            throw new IllegalArgumentException("Error: outputFile, timeoutPerFile required.");
        }
        return scriptArgs;
    }

    /**
     * Get the name of this file, excluding the extension, i.e., the SHA-256.
    */
    private String getProgramName() {
        String programName = currentProgram.getName();
        if (programName.contains(".")) {
            programName = programName.substring(0, programName.lastIndexOf('.'));
        }
        return programName;
    }

    /**
     * Check if the program is a PE file.
    */ 
    private boolean isPEFile() {
        Language language = currentProgram.getLanguage();
        String languageId = language.getLanguageID().getIdAsString().toLowerCase();
        return languageId.contains("x86") || languageId.contains("x64");
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

    /**
    * Refines the executable region by identifying code, data, and padding blocks.
    */
    private Regions getRegions(Listing listing, Address lower, Address upper) {
        List<Bounds> execBounds = new ArrayList<>();
        List<Bounds> codeBounds = new ArrayList<>();
        List<Bounds> dataBounds = new ArrayList<>();
        List<Bounds> paddBounds = new ArrayList<>();

        // The entire region is executable
        execBounds.add(
        new Bounds(
                virtualAddressToPhysicalAddress(lower),
                virtualAddressToPhysicalAddress(upper) + 1
            )
        );

        // If we're not doing the detailed analysis, just exit.
        if (!DETAILED_SECTIONS) {
            return new Regions(execBounds, codeBounds, dataBounds, paddBounds);
        }

        Address current = lower;
        Address regionStart = null;
        String currentType = null;
        long lowerPhysAddress;
        long upperPhysAddress;

        while (current.compareTo(upper) <= 0) {
            Instruction instruction = listing.getInstructionAt(current);
            boolean isData = listing.getDataAt(current) != null;
            String type;

            if (instruction != null) {
                type = "CODE";
                current = instruction.getMaxAddress().next();
            } else if (isData) {
                type = "DATA";
                current = current.next();
            } else {
                type = "PADD";
                current = current.next();
            }

            if (currentType == null || !currentType.equals(type)) {
                if (regionStart != null) {
                    lowerPhysAddress = virtualAddressToPhysicalAddress(regionStart);
                    upperPhysAddress = virtualAddressToPhysicalAddress(current.subtract(1));
                    Bounds bounds = new Bounds(lowerPhysAddress, upperPhysAddress + 1);
                    switch (currentType) {
                        case "CODE":
                            codeBounds.add(bounds);
                            break;
                        case "DATA":
                            dataBounds.add(bounds);
                            break;
                        case "PADD":
                            paddBounds.add(bounds);
                            break;
                    }
                }
                regionStart = current.subtract(1);
                currentType = type;
            }
        }

        // Handle the last region
        if (regionStart != null && currentType != null) {
            lowerPhysAddress = virtualAddressToPhysicalAddress(regionStart);
            upperPhysAddress = virtualAddressToPhysicalAddress(upper);
            Bounds bounds = new Bounds(lowerPhysAddress, upperPhysAddress + 1);
            switch (currentType) {
                case "CODE":
                    codeBounds.add(bounds);
                    break;
                case "DATA":
                    dataBounds.add(bounds);
                    break;
                case "PADD":
                    paddBounds.add(bounds);
                    break;
            }
        }

        return new Regions(execBounds, codeBounds, dataBounds, paddBounds);
    }

    /**
     * Convert regions to JSON string.
    */
    public static String regionsToJson(Regions regions) {
        StringBuilder jsonBuilder = new StringBuilder();
        jsonBuilder.append("{");

        jsonBuilder.append("\"EXEC\": ").append(boundsToJson(regions.execBounds)).append(", ");
        jsonBuilder.append("\"CODE\": ").append(boundsToJson(regions.codeBounds)).append(", ");
        jsonBuilder.append("\"DATA\": ").append(boundsToJson(regions.dataBounds)).append(", ");
        jsonBuilder.append("\"PADD\": ").append(boundsToJson(regions.paddBounds)).append("");

        jsonBuilder.append("}");
        return jsonBuilder.toString();
    }

    /**
     * Convert list of Bounds JSON string.
    */
    private static String boundsToJson(List<Bounds> boundsList) {
        StringBuilder arrayBuilder = new StringBuilder();
        arrayBuilder.append("[");

        for (int i = 0; i < boundsList.size(); i++) {
            Bounds bounds = boundsList.get(i);
            arrayBuilder.append("[").append(bounds.lower).append(", ").append(bounds.upper).append("]");

            if (i < boundsList.size() - 1) {
                arrayBuilder.append(", ");
            }
        }

        arrayBuilder.append("]");
        return arrayBuilder.toString();
    }

    /**
     * Helper struct that contains the boundaries of a region.
    */
    private static class Bounds {
        long lower;
        long upper;

        Bounds(long lower, long upper) {
            this.lower = lower;
            this.upper = upper;
        }
    }

    /**
     * Helper struct that contains the boundaries of the different types of regions.
    */
    private static class Regions {
        List<Bounds> execBounds;
        List<Bounds> codeBounds;
        List<Bounds> dataBounds;
        List<Bounds> paddBounds;

        Regions(List<Bounds> execBounds, List<Bounds> codeBounds,
                List<Bounds> dataBounds, List<Bounds> paddBounds) {
            this.execBounds = execBounds;
            this.codeBounds = codeBounds;
            this.dataBounds = dataBounds;
            this.paddBounds = paddBounds;
        }
    }
}
