import com.google.protobuf.MapEntry;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class G021HW1 {
    public static void main(String[] args) {
        //Commandline check
        if (args.length != 5) {
            throw new IllegalArgumentException("USAGE: filepath D M K L");
        }

        //Task 3 point 1 - Prints the command-line arguments and stores D,M,K,L into suitable variables.
        String file_path = args[0];
        float D = Float.parseFloat(args[1]);
        int M = Integer.parseInt(args[2]);
        int K = Integer.parseInt(args[3]);
        int L = Integer.parseInt(args[4]);

        System.out.println("File path: " + file_path + "D = " + D + " M = " + M + " K = " + K + " L = " + L);

        //Spark setup
        SparkConf conf = new SparkConf().setAppName("Outlier Detection").setMaster("local[*]");
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            //Reduce verbosity -- does not work somehow
            sc.setLogLevel("WARN");

            //Task 3  point 2 - Reads the input points into an RDD of strings (called rawData) and transform it into an RDD of points (called inputPoints), represented as pairs of floats, subdivided into L partitions.
            JavaRDD<String> rawData = sc.textFile(file_path);
            JavaPairRDD<Float, Float> inputPoints;

            //Conversion of string to a pair of points and storing in a new RDD
            inputPoints = rawData.mapToPair(line -> {
                String[] coordinates = line.split(",");
                float x_coord = Float.parseFloat(coordinates[0]);
                float y_coord = Float.parseFloat(coordinates[1]);

                return new Tuple2<>(x_coord, y_coord);
            });

            inputPoints.repartition(L).cache();

            //Task 3 point 3 - Prints the total number of points.
            long num_points = inputPoints.count();
            System.out.println("Number of points = " + num_points);

            //Task 3 point 4 - Only if the number of points is at most 200000:
            if (num_points <= 200000) {
                //Downloads the points into a list called listOfPoints
                List<Tuple2<Float, Float>> listOfPoints = inputPoints.collect();

                long stopwatch_start = System.currentTimeMillis();
                //Executes ExactOutliers with parameters listOfPoints,  D,M and K. The execution will print the information specified above.
                Methods.ExactOutliers(listOfPoints, D, M, K);
                long stopwatch_stop = System.currentTimeMillis();
                long exec_time = stopwatch_stop - stopwatch_start;
                //Prints ExactOutliers' running time. The stopwatch variable saves the current time when method starts and finishes
                System.out.println("Running time for ExactOutliers: " + exec_time + " millisec");
            }

            //Task 3 point 5 - In all cases:

            long stopwatch_startMR = System.currentTimeMillis();
            //Executes MRApproxOutliers with parameters inputPoints, D,M and K. The execution will print the information specified above.
            Methods.MRApproxOutliersBISCA1(inputPoints, D, M, K);
            long stopwatch_stopMR = System.currentTimeMillis();
            long exec_time = stopwatch_stopMR - stopwatch_startMR;
            //Prints MRApproxOutliers' running time. Again the stopwatch variable saves the current time when method starts and finishes
            System.out.println("Running time for MRApproxOutliers: " + exec_time + " millisec");

            stopwatch_startMR = System.currentTimeMillis();
            //Executes MRApproxOutliers with parameters inputPoints, D,M and K. The execution will print the information specified above.
            Methods.MRApproxOutliersBISCA2(inputPoints, D, M, K);
            stopwatch_stopMR = System.currentTimeMillis();
            exec_time = stopwatch_stopMR - stopwatch_startMR;
            //Prints MRApproxOutliers' running time. Again the stopwatch variable saves the current time when method starts and finishes
            System.out.println("Running time for MRApproxOutliers: " + exec_time + " millisec");

            sc.stop();
        }
    }
}

class Methods {
    private static float eucDistance(Tuple2<Float, Float> p1, Tuple2<Float, Float> p2) {
        float x_diff = p1._1 - p2._1;
        float y_diff = p1._2 - p2._2;

        return (float) Math.sqrt(Math.pow(x_diff, 2) + Math.pow(y_diff, 2));
    }

    private static Tuple2<Integer, Integer> determineCell(Tuple2<Float, Float> point, float D) {
        float lambda = (float) (D / (2 * Math.sqrt(2)));

        int i = (int) Math.floor(point._1 / lambda);
        int j = (int) Math.floor(point._2 / lambda);

        return new Tuple2<>(i, j);
    }

    public static void ExactOutliers(List<Tuple2<Float, Float>> points, float D, int M, int K) {

        ArrayList<Tuple2<Integer, Integer>> outliersPoints = new ArrayList<>();

        for (int i = 0; i < points.size(); i++) {
            int dNeighborCountI = 0;
            for (Tuple2<Float, Float> point : points) {
                //if(i==j) continue;
                if (eucDistance(points.get(i), point) <= D) dNeighborCountI++;
            }
            if (dNeighborCountI <= M) outliersPoints.add(new Tuple2<>(i, dNeighborCountI));
        }

        //Sorting the list of outliers points by dNeighborCount(|B(p,D)|)
        outliersPoints.sort((e1, e2) -> e1._2().compareTo(e2._2));

        System.out.println("Number of (" + D + "," + M + ")-outliers: " + outliersPoints.size());
        for (int i = 0; i < (Math.min(outliersPoints.size(), K)); i++) {
            //The first min(size of outliersPoints list,K) outliers points
            System.out.println(points.get(outliersPoints.get(i)._1));
        }
    }

    public static void MRApproxOutliersBISCA1(JavaPairRDD<Float, Float> points, float D, int M, int K) {
        //step a
        float lambda = (float) (D / (2 * Math.sqrt(2)));

        JavaPairRDD<Tuple2<Integer, Integer>, Integer> cellCount = points.flatMapToPair(
                (pair) -> {
                    ArrayList<Tuple2<Tuple2<Integer, Integer>, Integer>> pairsList = new ArrayList<>();
                    pairsList.add(new Tuple2<>(new Tuple2<>((int) Math.floor(pair._1 / lambda), (int) Math.floor(pair._2 / lambda)), 1));

                    return pairsList.iterator();
                }
        ).reduceByKey(Integer::sum);


        //step b -- Semi iterative algorithm
        Map<Tuple2<Integer, Integer>, Integer> tmpMap = cellCount.collectAsMap();
        JavaPairRDD<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>> outliersPoint = cellCount.flatMapToPair(
                (pair) -> {

                    HashMap<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>> pairSizeN3N7 = new HashMap<>();
                    pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, 0, 0));

                    //Notice: this for count itself too
                    for (int i = -3; i < 4; i++) {
                        for (int j = -3; j < 4; j++) {
                            if (tmpMap.get(new Tuple2<>(pair._1._1 + i, pair._1._2 + j)) != null) {
                                int cellIJCount = tmpMap.get(new Tuple2<>(pair._1._1 + i, pair._1._2 + j));
                                if ((i < -1 || i > 1) || (j < -1 || j > 1))
                                    //In region C7
                                    pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, pairSizeN3N7.get(pair._1)._2(), pairSizeN3N7.get(pair._1)._3() + cellIJCount));
                                else
                                    //In region C3
                                    pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, pairSizeN3N7.get(pair._1)._2() + cellIJCount, pairSizeN3N7.get(pair._1)._3() + cellIJCount));
                            }
                        }
                    }

                    ArrayList<Tuple2<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>>> compPoints = new ArrayList<>();
                    for (Map.Entry<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>> e : pairSizeN3N7.entrySet()) {
                        compPoints.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return compPoints.iterator();
                }
        );

        List<Tuple2<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>>> finalList = outliersPoint.collect();
        int outliers = 0, uncertains = 0;
        for (Tuple2<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>> elem : finalList) {
            if (elem._2._3() <= M) outliers += elem._2._1();
            if (elem._2._2() <= M && elem._2._3() > M) uncertains += elem._2._1();
        }
        System.out.println("Number of sure outliers = " + outliers);
        System.out.println("Number of uncertain points = " + uncertains);

        JavaPairRDD<Integer, Tuple2<Integer, Integer>> ordercell = cellCount.flatMapToPair(
                (pair) -> {
                    ArrayList<Tuple2<Integer, Tuple2<Integer, Integer>>> pairsList = new ArrayList<>();
                    pairsList.add(new Tuple2<>(pair._2, new Tuple2<>(pair._1._1(), pair._1._2())));

                    return pairsList.iterator();
                }
        );

        List<Tuple2<Integer, Tuple2<Integer, Integer>>> firstKElements = ordercell.sortByKey().take(K);
        for (Tuple2<Integer, Tuple2<Integer, Integer>> e : firstKElements) {
            System.out.println("Cell: " + e._2 + " Size=" + e._1);
        }
    }
    public static void MRApproxOutliersBISCA2(JavaPairRDD<Float, Float> points, float D, int M, int K) {
        //step a
        float lambda = (float) (D / (2 * Math.sqrt(2)));

        JavaPairRDD<Tuple2<Integer, Integer>, Integer> cellCount = points.flatMapToPair(
                (pair) -> {
                    ArrayList<Tuple2<Tuple2<Integer, Integer>, Integer>> pairsList = new ArrayList<>();
                    pairsList.add(new Tuple2<>(new Tuple2<>((int) Math.floor(pair._1 / lambda), (int) Math.floor(pair._2 / lambda)), 1));

                    return pairsList.iterator();
                }
        ).reduceByKey(Integer::sum);


        //step b -- Iterative algorithm
        Map<Tuple2<Integer, Integer>, Integer> tmpMap = cellCount.collectAsMap();
        HashMap<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>> pairSizeN3N7 = new HashMap<>();

        for(Map.Entry<Tuple2<Integer, Integer>, Integer> e : tmpMap.entrySet()){
            Tuple2<Tuple2<Integer,Integer>,Integer> pair =  new Tuple2<>(e.getKey(),e.getValue());

            pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, 0, 0));
                    //Notice: this for count itself too
                    for (int i = -3; i < 4; i++) {
                        for (int j = -3; j < 4; j++) {
                            if (tmpMap.get(new Tuple2<>(pair._1._1 + i, pair._1._2 + j)) != null) {
                                int cellIJCount = tmpMap.get(new Tuple2<>(pair._1._1 + i, pair._1._2 + j));
                                if ((i < -1 || i > 1) || (j < -1 || j > 1))
                                    //In region C7
                                    pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, pairSizeN3N7.get(pair._1)._2(), pairSizeN3N7.get(pair._1)._3() + cellIJCount));
                                else
                                    //In region C3
                                    pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, pairSizeN3N7.get(pair._1)._2() + cellIJCount, pairSizeN3N7.get(pair._1)._3() + cellIJCount));
                            }
                        }
                    }
        }

        int outliers = 0, uncertains = 0;
        for (Map.Entry<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Integer>> elem : pairSizeN3N7.entrySet()) {
            if (elem.getValue()._3() <= M) outliers += elem.getValue()._1();
            if (elem.getValue()._2() <= M && elem.getValue()._3() > M) uncertains += elem.getValue()._1();
        }
        System.out.println("Number of sure outliers = " + outliers);
        System.out.println("Number of uncertain points = " + uncertains);

        JavaPairRDD<Integer, Tuple2<Integer, Integer>> ordercell = cellCount.flatMapToPair(
                (pair) -> {
                    ArrayList<Tuple2<Integer, Tuple2<Integer, Integer>>> pairsList = new ArrayList<>();
                    pairsList.add(new Tuple2<>(pair._2, new Tuple2<>(pair._1._1(), pair._1._2())));

                    return pairsList.iterator();
                }
        );

        List<Tuple2<Integer, Tuple2<Integer, Integer>>> firstKElements = ordercell.sortByKey().take(K);
        for (Tuple2<Integer, Tuple2<Integer, Integer>> e : firstKElements) {
            System.out.println("Cell: " + e._2 + " Size=" + e._1);
        }
    }

}