package ensemblelearning;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;

/**
 *
 * @author Pavitra
 */
class DecisionTree implements MyClassifier {

    @Override
    public void runClassifier() throws Exception {
        // train classifier
        Classifier cls = new J48();
        cls.buildClassifier(Driver.TRAINING_DATA);
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(Driver.TRAINING_DATA);
        eval.evaluateModel(cls, Driver.TEST_DATA);
        System.out.println(eval.toSummaryString("\nResults: Decision Tree\n=====================\n", false));
    }
}
