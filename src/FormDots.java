import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

public class FormDots extends JFrame implements Runnable, MouseListener, KeyListener {

    private final int width = 1280;
    private final int height = 720;

    private BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    private BufferedImage pimg = new BufferedImage(width /8, height /8, BufferedImage.TYPE_INT_RGB);
    private int frame;

    private NeuralNetwork nNet;

    public List<Point> points = new ArrayList<>();

    public FormDots() {
        UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
        UnaryOperator<Double> derSigmoid = y -> y * (1 - y);

        nNet = new NeuralNetwork(0.001, sigmoid, derSigmoid, 2, 5, 5, 2);

        this.setSize(width + 16, height + 38);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocation(50, 50);
        this.add(new JLabel(new ImageIcon(img)));
        addMouseListener(this);
    }

    @Override
    public void run() {
        while (true) {
            this.repaint();
        }
    }

    @Override
    public void paint(Graphics g) {
        if (points.size() > 0) {
            for (int k = 0; k < 10000; k++) {
                Point p = points.get((int) (Math.random() * points.size()));
                double nx = (double) p.x / width - 0.5;
                double ny = (double) p.y / height - 0.5;

                nNet.feedForward(new double[]{nx, ny});
                double[] targets = new double[2];
                if (p.type == 0) targets[0] = 1;
                else targets[1] = 1;
                nNet.backpropagation(targets);
            }
        }

        for (int i = 0; i < width / 8; i++) {
            for (int j = 0; j < height / 8; j++) {
                double nx = (double) i / width * 8 - 0.5;
                double ny = (double) j / height * 8 - 0.5;
                double[] outputs = nNet.feedForward(new double[]{nx, ny});
                double green = Math.max(0, Math.min(1, outputs[0] - outputs[1]));
                double blue = 1 - green;
                green = 0.3 + green * 0.5;
                blue = 0.5 + blue * 0.5;

                int color = (100 << 16) | ((int) (green * 255) << 8) | (int) (blue * 255);
                pimg.setRGB(i, j, color);
            }
        }

        Graphics graphics = img.getGraphics();
        graphics.drawImage(pimg, 0, 0, width, height, this);

        for (Point p : points) {
            graphics.setColor(Color.WHITE);
            graphics.fillOval(p.x - 3, p.y - 3, 26, 26);
            if (p.type == 0) graphics.setColor(Color.GREEN);
            else graphics.setColor(Color.BLUE);
            graphics.fillOval(p.x,p.y, 20,20);
        }
        g.drawImage(img, 8, 30, width, height, this);
        frame++;
    }

    @Override
    public void mouseClicked(MouseEvent e){

    }

    @Override
    public void mousePressed(MouseEvent e){
        int type = 0;
        if(e.getButton() == 3) type = 1;
        int x = e.getX();
        int y = e.getY();
        points.add(new Point(x-16, y - 38, type));
    }

    @Override
    public void mouseReleased(MouseEvent e){

    }

    @Override
    public void mouseEntered(MouseEvent e){

    }

    @Override
    public void mouseExited(MouseEvent e){

    }

    @Override
    public void keyTyped(KeyEvent e) {

    }

    @Override
    public void keyPressed(KeyEvent e) {
        if(e.getKeyCode() == KeyEvent.VK_SPACE){
            
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {

    }
}