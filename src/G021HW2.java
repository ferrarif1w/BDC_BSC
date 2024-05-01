import java.util.*;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;
import scala.Tuple3;

public class G021HW2 {
    public static void main(String[] args) {
        //Commandline check
        if(args.length != 4){
            throw new IllegalArgumentException("USAGE: filepath M K L");
        }

        //Task 3 point 1 - Prints the command-line arguments and stores M,K,L into suitable variables.
        String file_path = args[0];
        int M = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);

        System.out.println(file_path + " M=" + M + " K=" + K + " L=" + L);

        //Spark setup

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("Outlier Detection V2").setMaster("local[*]");
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            //Reduce verbosity -- does not work somehow
            sc.setLogLevel("WARN");

            //Task 3 point 2 - Reads the input points into an RDD of strings (called rawData) and transform it into an RDD of points (called inputPoints), represented as pairs of floats, subdivided into L partitions.
            JavaRDD<String> rawData = sc.textFile(file_path);

            //Conversion of string to a pair of points and storing in a new RDD
            JavaRDD<Tuple2<Float, Float>> inputPoints = rawData.map(line -> {
                String[] coordinates = line.split(",");
                float x_coord = Float.parseFloat(coordinates[0]);
                float y_coord = Float.parseFloat(coordinates[1]);

                return new Tuple2<>(x_coord, y_coord);
            });

            inputPoints.repartition(L).cache();

            //Task 3 point 3 - Prints the total number of points.
            long num_points = inputPoints.count();
            System.out.println("Number of points = " + num_points);

            //Executes MRFFT with parameters inputPoints and K and stores the returned radius into a float D.
            float D = MethodsHW2.MRFFT(inputPoints,K,sc);

            //Executes MRApproxOutliers, modified as described above, with parameters inputPoints, D,M.
            MethodsHW2.MRApproxOutliers(inputPoints, D,M);

            sc.stop();
        }
    }
}

class MethodsHW2{
    private static float eucDistance(Tuple2<Float,Float> p1, Tuple2<Float,Float> p2){
        float x_diff = p1._1 - p2._1;
        float y_diff = p1._2 - p2._2;

        return (float) Math.sqrt(Math.pow(x_diff,2)+Math.pow(y_diff,2));
    }

    private static Tuple2<Integer, Integer> determineCell(Tuple2<Float, Float> point, float D) {
        float lambda = (float) (D/(2*Math.sqrt(2)));

        int i = (int) Math.floor(point._1/lambda);
        int j = (int) Math.floor(point._2/lambda);

        return new Tuple2<>(i, j);
    }

    public static void MRApproxOutliers(JavaRDD<Tuple2<Float, Float>> points, float D, int M) {

        JavaPairRDD<Tuple2<Integer, Integer>, Long> cellCount = points.mapToPair(
                (pair) -> new Tuple2<>(determineCell(pair, D), 1L)
        ).reduceByKey(Long::sum);

        Map<Tuple2<Integer, Integer>, Long> tmpMap = cellCount.collectAsMap();
        HashMap<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> pairSizeN3N7 = new HashMap<>();

        for(Map.Entry<Tuple2<Integer, Integer>, Long> e : tmpMap.entrySet()){
            Tuple2<Tuple2<Integer,Integer>,Long> pair =  new Tuple2<>(e.getKey(),e.getValue());

            pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, 0L, 0L));

            for (int i = -3; i < 4; i++) {
                for (int j = -3; j < 4; j++) {
                    if (tmpMap.get(new Tuple2<>(pair._1._1 + i, pair._1._2 + j)) != null) {
                        long cellIJCount = tmpMap.get(new Tuple2<>(pair._1._1 + i, pair._1._2 + j));
                        if ((i < -1 || i > 1) || (j < -1 || j > 1))
                            pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, pairSizeN3N7.get(pair._1)._2(), pairSizeN3N7.get(pair._1)._3() + cellIJCount));
                        else
                            pairSizeN3N7.put(pair._1, new Tuple3<>(pair._2, pairSizeN3N7.get(pair._1)._2() + cellIJCount, pairSizeN3N7.get(pair._1)._3() + cellIJCount));
                    }
                }
            }
        }

        int outliers = 0, uncertains = 0;
        for (Map.Entry<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> elem : pairSizeN3N7.entrySet()) {
            if (elem.getValue()._3() <= M) outliers += elem.getValue()._1();
            if (elem.getValue()._2() <= M && elem.getValue()._3() > M) uncertains += elem.getValue()._1();
        }
        System.out.println("Number of sure outliers = " + outliers);
        System.out.println("Number of uncertain points = " + uncertains);
    }

    private static Tuple2<Integer,Float> findFarthestPoint(ArrayList<Tuple2<Float,Float>> points,ArrayList<Tuple2<Float,Float>> centers){
        //TODO: I'm so bad to explain my code, please help me! :<3
        //Tmp ArrayList which stores the index of each point in points and the distance from the closest center in centers
        ArrayList<Tuple2<Integer,Float>> indexPointDS = new ArrayList<>();

        for(int i = 0; i < points.size(); i++){
            float distFromS = Float.MAX_VALUE; //Min distance from point points[i] to the set of centers S
            for(Tuple2<Float,Float> center: centers)
                if (eucDistance(points.get(i), center) < distFromS) distFromS = eucDistance(points.get(i), center);
            indexPointDS.add(new Tuple2<>(i,distFromS));
        }

        return Collections.max(indexPointDS, (e1, e2) -> e1._2().compareTo(e2._2));
        //._1 --> index (in the ArrayList points) of the farthest point from the set of center S
        //._2 --> distance of the farthest point from the set of center S
    }

    private static ArrayList<Tuple2<Float,Float>> SequentialFFT(ArrayList<Tuple2<Float,Float>> points, int K){
        ArrayList<Tuple2<Float,Float>> centers = new ArrayList<>();
        centers.add(points.remove(0));

        for(int i=1;i<K;i++){
            int newCenterIndex = findFarthestPoint(points,centers)._1;
            centers.add(points.remove(newCenterIndex));
        }

        return centers;
    }

    public static float MRFFT(JavaRDD<Tuple2<Float, Float>> points, int K, JavaSparkContext sc){


        long stopwatch_startRound1 = System.currentTimeMillis();
        //Round 1
        List<Tuple2<Float,Float>> coreset = points.mapPartitions(
                (partition) -> {
                    ArrayList<Tuple2<Float,Float>> partitionPoints = new ArrayList<>();
                    partition.forEachRemaining(partitionPoints::add);

                    return SequentialFFT(partitionPoints,K).iterator();
                }
        ).collect();
        System.out.println("Running time of Round 1 MRFFT: " + (System.currentTimeMillis() - stopwatch_startRound1) + " ms");


        long stopwatch_startRound2= System.currentTimeMillis();
        //Round 2
        Broadcast<ArrayList<Tuple2<Float, Float>>> kCenters = sc.broadcast(SequentialFFT(new ArrayList<>(coreset),K));
        System.out.println("Running time of Round 2 MRFFT: " + (System.currentTimeMillis() - stopwatch_startRound2) + " ms");


        long stopwatch_startRound3 = System.currentTimeMillis();
        //Round 3
        float radius = points.map(
                    (point) -> findFarthestPoint(new ArrayList<>(Collections.singletonList(point)), kCenters.value())._2
                ).reduce(Float::max);
        System.out.println("Running time of Round 3 MRFFT: " + (System.currentTimeMillis() - stopwatch_startRound3) + " ms");

        return radius;
    }
}