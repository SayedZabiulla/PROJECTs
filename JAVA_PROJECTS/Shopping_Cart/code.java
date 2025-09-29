package JAVA_PROJECTS.Shopping_Cart;
import java.util.Scanner;

public class code {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String item;
        double price;
        int quantity;
        char currency = '$';
        double total;

        System.out.print("what item would you like to buy?: ");
        item =sc.nextLine();

        System.out.print("What is the price for each?: ");
        price =sc.nextDouble();

        System.out.print("How many would you like to buy?: ");
        quantity =sc.nextInt();

        total = price*quantity;

        System.out.println("\nYou Have bought "+quantity+" "+item+"/s");
        System.out.println("Your Total is "+currency+total);

        sc.close();
    }
}
