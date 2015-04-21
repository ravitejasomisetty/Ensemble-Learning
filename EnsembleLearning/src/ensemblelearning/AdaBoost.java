package ensemblelearning;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

class AdaBoost implements MyClassifier {

    @Override
    public void runClassifier() throws Exception {
        AdaBoostM1 m1 = new AdaBoostM1();
        m1.setClassifier(new J48());
        m1.buildClassifier(Driver.TRAINING_DATA);

        Evaluation em1 = new Evaluation(Driver.TRAINING_DATA);
        em1.evaluateModel(m1, Driver.TEST_DATA);

        String summarynaive = em1.toSummaryString("\nResults: AdaBoost\n======================\n", false);
        System.out.println(summarynaive);
    }
}
