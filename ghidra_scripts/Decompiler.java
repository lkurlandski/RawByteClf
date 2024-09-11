import java.io.FileWriter;
import java.io.IOException;
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
    protected void processFunction(Function func, FileWriter writer) throws Exception {
        String decompiledCode = decompileFunction(func);
        writer.write(decompiledCode);
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
                decompiledCode = replaceSignature(decompiledFunc, decompiledCode, signature);
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

    private String replaceSignature(DecompiledFunction decompiledFunc, String decompiledCode, String signature) throws Exception {
        String signatureCur = decompiledFunc.getSignature();
        signatureCur = signatureCur.substring(0, signatureCur.indexOf(";"));
        if (signatureCur.contains("\n")) {
            throw new Exception("Anomalous function signature detected.");
        }

        if (decompiledCode.contains(signatureCur)) {
            String tmpPattern = Pattern.quote(signatureCur);
            return decompiledCode.replaceFirst(tmpPattern, signature);
        }

        validateDecompiledCode(decompiledCode);

        String argumentsTall = decompiledCode.substring(decompiledCode.indexOf("("), decompiledCode.indexOf(")") + 1);
        String argumentsFlat = signatureCur.substring(signatureCur.indexOf("("), signatureCur.indexOf(")") + 1);
        
        decompiledCode = decompiledCode.replaceFirst(Pattern.quote(argumentsTall), argumentsFlat);
        return decompiledCode.replaceFirst(Pattern.quote(signatureCur), signature);
    }

    private void validateDecompiledCode(String decompiledCode) throws Exception {
        boolean encounteredOpenParenthesis = false;
        boolean encounteredCloseParenthesis = false;
        for (int i = 0; i < decompiledCode.length(); i++) {
            char currentChar = decompiledCode.charAt(i);
            if (currentChar == '(') {
                if (encounteredOpenParenthesis) {
                    throw new Exception("Anomalous function signature detected.");
                }
                encounteredOpenParenthesis = true;
            } else if (currentChar == ')') {
                if (encounteredCloseParenthesis) {
                    throw new Exception("Anomalous function signature detected.");
                }
                encounteredCloseParenthesis = true;
            } else if (currentChar == '{') {
                if (!(encounteredOpenParenthesis && encounteredCloseParenthesis)) {    
                    throw new Exception("Anomalous function signature detected.");
                }
                break;
            } else if (i == decompiledCode.length() - 1) {
                throw new Exception("Anomalous function signature detected.");
            }
        }
    }
}
