import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.TensorFlow;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

public class Main {





    public static void main(String[] args) throws IOException {
        System.out.println(TensorFlow.version());
        nu.pattern.OpenCV.loadLocally();

        TensorFlowDetector detector = new TensorFlowDetector("./lib/model");

        BufferedImage image = ImageIO.read(new File("./data/family.jpg"));
        WritableRaster raster = image.getRaster();
        DataBufferByte data   = (DataBufferByte) raster.getDataBuffer();

        BufferedImage faceDetectedImage = detector.process(image.getWidth(),image.getHeight(),image);
        File outputfile = new File("./data/family_blured.jpg");
        ImageIO.write(faceDetectedImage, "jpg", outputfile);



    }


}
