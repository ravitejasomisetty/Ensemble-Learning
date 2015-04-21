package ensemblelearning;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;

/**
 *
 * @author RavitejaSomisetty
 */
public class Driver {

    /**
     * @param args the command line arguments
     */
    static String TRAINING_FILE = "E:\\Spring2015\\Data Mining\\Ensemble Learning Project\\Ensemble-Learning\\census-income.data";
    static String TEST_FILE = "E:\\Spring2015\\Data Mining\\Ensemble Learning Project\\Ensemble-Learning\\census-income.test";
    static Instances TRAINING_DATA, TEST_DATA;

    public static void main(String[] args) throws IOException {

        PreProcessor pTrain = new PreProcessor(TRAINING_FILE);
        pTrain.run();
        TRAINING_DATA = pTrain.getData();

        PreProcessor pTest = new PreProcessor(TEST_FILE);
        pTest.run();
        TEST_DATA = pTest.getData();

        try {
            new NaiveBayesClassifier().runClassifier();
        } catch (Exception ex) {
            Logger.getLogger(Driver.class.getName()).log(Level.SEVERE, null, ex);
        }

        try {
            new DecisionTree().runClassifier();
        } catch (Exception ex) {
            Logger.getLogger(Driver.class.getName()).log(Level.SEVERE, null, ex);
        }

        try {
            new AdaBoost().runClassifier();
        } catch (Exception ex) {
            Logger.getLogger(Driver.class.getName()).log(Level.SEVERE, null, ex);
        }

        try {
            new BaggingMethod().runClassifier();
        } catch (Exception ex) {
            Logger.getLogger(Driver.class.getName()).log(Level.SEVERE, null, ex);
        }
        try {
            new SVM().runClassifier();
        } catch (Exception ex) {
            Logger.getLogger(Driver.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
