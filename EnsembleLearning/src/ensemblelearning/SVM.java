/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ensemblelearning;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;

/**
 *
 * @author RavitejaSomisetty
 */
public class SVM implements MyClassifier {

    @Override
    public void runClassifier() throws Exception {
        SMO smo = new SMO();
        smo.setOptions(weka.core.Utils.splitOptions("-C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
        smo.buildClassifier(Driver.TRAINING_DATA);
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(Driver.TRAINING_DATA);
        eval.evaluateModel(smo, Driver.TEST_DATA);
        System.out.println(eval.toSummaryString("\nResults: SVM\n================\n", false));
    }

}
