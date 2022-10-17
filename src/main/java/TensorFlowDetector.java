
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.geom.Rectangle2D;
import java.awt.image.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;



public class TensorFlowDetector implements IDeepLearningProcessor {

	private Classifier classifier;
	private String streamId;
	private long captureCount = 0;

	private List<Classifier.Recognition> recognitionList = new ArrayList<Classifier.Recognition>();
	private long lastUpdate;
	private boolean tensorflowRunning;


	public TensorFlowDetector(String modelDir) throws IOException {
		this.classifier = TFObjectDetector.create(modelDir);
	}


	@Override
	public BufferedImage process(int width, int height, BufferedImage image) throws IOException {
		long startTime = System.currentTimeMillis();

		//BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Rectangle myRectangle = null;

		/*int k = 0;
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				int r = (int)(data[k++]& 0xFF);
				int g = (int)(data[k++]& 0xFF);
				int b = (int)(data[k++]& 0xFF);
				//int a = (int)(data[k++]& 0xFF);

				Color c = new Color(r, g, b);
				image.setRGB(x, y, c.getRGB());
			}
		}*/
		ColorModel cm = image.getColorModel();
		boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
		WritableRaster raster = image.copyData(null);
		BufferedImage copyImage = new BufferedImage(cm, raster, isAlphaPremultiplied, null);
		recognitionList = classifier.recognizeImage(copyImage);
		ArrayList<Rectangle> rectList = new ArrayList<>();
		if (recognitionList.size() > 0) {
			for (Classifier.Recognition recognition : recognitionList) {
				 double rectangleX = recognition.getLocation().getMinX()-10;
				 double rectangleY = recognition.getLocation().getMinY()-10;
				 double rectangleWidth = recognition.getLocation().getWidth()+20;
				 double rectangleHeight = recognition.getLocation().getHeight()+20;
				 if(rectangleX < 0){
					 rectangleX = 0;
				 }
				 if(rectangleY < 0){
					 rectangleY = 0;
				 }
				 if(rectangleX + rectangleWidth > image.getWidth()){
					 rectangleX = image.getWidth() - rectangleWidth;
				 }
				if(rectangleY + rectangleHeight > image.getHeight()){
					rectangleY = image.getHeight() - rectangleHeight;
				}

				Rectangle rectangle = new Rectangle((int) rectangleX,
						(int) rectangleY,
						(int) rectangleWidth,
						(int) rectangleHeight);
				rectList.add(rectangle);

				
				//g2D.drawString(text, (int)recognition.getLocation().getMinX(), (int) recognition.getLocation().getMinY());
			}

			captureCount++;
		}
		return blurRectangles(image,rectList,null);

		//return image;
	}


	BufferedImage blurRectangles(BufferedImage img,ArrayList<Rectangle> rectList,BluringTechnique bluringTechnique) throws IOException {
		if(bluringTechnique == BluringTechnique.CONVOLUTION_BLUR){
			float weight = 1.0f / (25 * 25);
			float[] data = new float[25 * 25];
			Arrays.fill(data, weight);
			for(Rectangle rectangle:rectList){
				BufferedImage dest = img.getSubimage(rectangle.x, rectangle.y, rectangle.width-2, rectangle.height-2); // x, y, width, height
				ColorModel cm = img.getColorModel();
				BufferedImage src = new BufferedImage(cm, dest.copyData(dest.getRaster().createCompatibleWritableRaster()), cm.isAlphaPremultiplied(), null).getSubimage(0, 0, dest.getWidth(), dest.getHeight());
				Kernel kernel = new Kernel(25, 25, data);
				new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null).filter(src, dest);
			}
		}else{
			for(Rectangle rectangle:rectList){

				Rect rect = new Rect(rectangle.x,rectangle.y,rectangle.width,rectangle.height);
				Mat srcMat = Utils.BufferedImage2Mat(img);

				Imgproc.GaussianBlur(new Mat(srcMat,rect),new Mat(srcMat,rect), new Size(15,15),0);
				img = Utils.Mat2BufferedImage(srcMat);

			}

		}

		return img;
	}
}
