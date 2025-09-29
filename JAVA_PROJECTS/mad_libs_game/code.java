import java.util.Scanner;

public class code {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String adj1, adj2;
        String noun1, noun2, noun3, noun4, noun5, noun6, noun7, noun8;
        String verb1, verb2, verb3, verb4;


        System.out.print("An adjective (Time): ");
        adj1 = sc.nextLine();
        System.out.print("An adjective (Emotion): ");
        adj2 = sc.nextLine();

        System.out.print("A noun: ");
        noun1 = sc.nextLine();
        System.out.print("A noun: ");
        noun2 = sc.nextLine();
        System.out.print("A noun: ");
        noun3 = sc.nextLine();
        System.out.print("A noun: ");
        noun4 = sc.nextLine();
        System.out.print("A noun: ");
        noun5 = sc.nextLine();
        System.out.print("A noun: ");
        noun6 = sc.nextLine();
        System.out.print("A noun (plural): ");
        noun7 = sc.nextLine();
        System.out.print("A noun: ");
        noun8 = sc.nextLine();

        // Get verbs
        System.out.print("A verb (past tense): ");
        verb1 = sc.nextLine();
        System.out.print("Another verb (present tense): ");
        verb2 = sc.nextLine();
        System.out.print("A verb ending in -ing: ");
        verb3 = sc.nextLine();
        System.out.print("Another verb ending in -ing: ");
        verb4 = sc.nextLine();

        System.out.println();
        System.out.println("It was a " + adj1 + " cold November day.");
        System.out.println("I woke up to the " + adj2 + " smell of " + noun1 +" roasting in the " + noun2 + " downstairs.");
        System.out.println("I " + verb1 + " down the stairs to see if I could help " +verb2 + " the dinner.");
        System.out.println("See if " + noun3 + " needs a fresh " + noun4 + ".");
        System.out.println("So I carried a tray of glasses full of " + noun5 +" into the " + verb3 + " room.");
        System.out.println("When I got there, I couldn't believe my " + noun6 + "!");
        System.out.println("There were " + noun7 + " " + verb4 + " on the " + noun8 + "!");

        sc.close();
    }
}