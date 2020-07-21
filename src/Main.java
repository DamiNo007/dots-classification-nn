public class Main {

    public static void main(String[] args){
        dots();
    }

    private static void dots(){
        FormDots f = new FormDots();
        new Thread(f).start();
    }
}
