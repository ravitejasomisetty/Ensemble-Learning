/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ensemblelearning;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;

/**
 *
 * @author RavitejaSomisetty
 */
public class KNNClassifier implements MyClassifier {

    @Override
    public void runClassifier() throws Exception {
        Classifier c = new IBk();
        c.buildClassifier(Driver.TRAINING_DATA);

        Evaluation e = new Evaluation(Driver.TRAINING_DATA);
        e.evaluateModel(c, Driver.TEST_DATA);
        String summarynaive = e.toSummaryString("\nResults: KNN \n======================\n", false);
        System.out.println(summarynaive);
    }

}
