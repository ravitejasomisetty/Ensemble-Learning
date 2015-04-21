package ensemblelearning;


import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;

class BaggingMethod implements MyClassifier {

    @Override
    public void runClassifier() throws Exception {
        Bagging b = new Bagging();
        b.setClassifier(new J48());
        b.buildClassifier(Driver.TRAINING_DATA);

        Evaluation eb = new Evaluation(Driver.TRAINING_DATA);
        eb.evaluateModel(b, Driver.TEST_DATA);

        String summarynaive = eb.toSummaryString("\nResults: Bagging\n======================\n", false);
        System.out.println(summarynaive);
    }
}
