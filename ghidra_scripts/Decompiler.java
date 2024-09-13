/**
 * Lift binaries to decompiled C-like code.
*/

import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ghidra.app.decompiler.DecompiledFunction;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.listing.Function;

public class Decompiler extends Lifter {

    private DecompInterface decompInterface;

    @Override
    protected void run() throws Exception {
        decompInterface = new DecompInterface();
        decompInterface.openProgram(getCurrentProgram());
        try {
            super.run();
        } finally {
            decompInterface.dispose();
        }
    }

    @Override
    protected String processFunction(Function func) throws Exception {
        return decompileFunction(func);
    }

    @Override
    protected String getFileExtension() {
        return ".c";
    }

    private String decompileFunction(Function func) throws Exception {
        DecompileResults results = decompInterface.decompileFunction(func, this.timeoutPerFunc, null);
        String signature = func.getPrototypeString(FORMAL_SIGNATURE, INCLUDE_CALLING_CONVENTION);

        if (results.decompileCompleted()) {
            DecompiledFunction decompiledFunc = results.getDecompiledFunction();
            String decompiledCode = decompiledFunc.getC();
            if (REPLACE_SIGNATURE) {
                decompiledCode = replaceSignature(decompiledCode, signature);
            }
            return decompiledCode;
        }

        String message;
        if (results.isTimedOut()) {
            println("decompileFunction: results.isTimedOut()=true");
            message = "WARNING: Decompilation incomplete due to timeoutPerFile=" + this.timeoutPerFile + ".";
        } else if (results.isCancelled()) {
            println("decompileFunction: results.isCancelled()=true");
            message = "WARNING: Decompilation incomplete due to timeoutPerFunc=" + this.timeoutPerFunc + ".";
        } else {
            println("decompileFunction: results.getErrorMessage()=" + results.getErrorMessage().replace("\n", ""));
            message = "WARNING: Decompilation incomplete due to ErrorMessage=`" + results.getErrorMessage().replace("\n", "") + "`.";
        }

        return signature + "\n{\n\n/* " + message + " */\n}\n\n";
    }

    public String replaceSignature(String decompiledCode, String newSignature) throws Exception {
        // Define a regex pattern that captures a function signature. It allows for varying spaces/newlines.
        String functionSignaturePattern = "(?s)(\\b[\\w\\s*]+\\s+)?(\\w+\\s*\\(.*?\\))";

        // Compile the regex pattern
        Pattern pattern = Pattern.compile(functionSignaturePattern);
        Matcher matcher = pattern.matcher(decompiledCode);

        // Find the first match and replace it with the new signature
        if (matcher.find()) {
            String originalSignature = matcher.group(1);
            return matcher.replaceFirst(Matcher.quoteReplacement(newSignature));
        }
        println("replaceSignature: newSignature=" + newSignature);
        println("replaceSignature: decompiledCode=" + decompiledCode);
        throw new Exception("Failed to replace the signature.");
    }
}
