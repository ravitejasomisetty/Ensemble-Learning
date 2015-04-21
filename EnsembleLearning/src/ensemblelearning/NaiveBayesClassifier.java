package ensemblelearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 *
 * @author Pavitra
 */
class NaiveBayesClassifier implements MyClassifier {

    @Override
    public void runClassifier() throws Exception {
        Classifier cnaive = (Classifier) new NaiveBayes();
        cnaive.buildClassifier(Driver.TRAINING_DATA);

        Evaluation enaive = new Evaluation(Driver.TRAINING_DATA);
        enaive.evaluateModel(cnaive, Driver.TEST_DATA);

        String summarynaive = enaive.toSummaryString("\nResults: Naive Bayes\n======================\n", false);
        System.out.println(summarynaive);
    }
}
