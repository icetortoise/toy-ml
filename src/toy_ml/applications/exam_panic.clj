(ns toy-ml.applications.exam-panic
  (:use [toy-ml core]
        [incanter core charts distributions]
        [clojure.core.match :only [match]])
  (:require [incanter.stats :as stats]))

(def pb [0.5, 0.5])
(def pr-given-b [[0.3 0.7][0.8 0.2]])
(def pa-given-b [[0.1 0.9][0.5 0.5]])
(def pp-given-ra [[0 1][0.8 0.2][0.6,0.4][1,0]])

(defn pb-given-rap
  "
P(b|rap)=P(b|ra)=P(ra|b)*P(b)/P(ra)=P(r|b)*P(a|b)*P(b)/P(ra)
r a    P(b)
T T    0.3*0.1*0.5/0.215=0.0698
T F    0.3*0.9*0.5/0.335=0.4030
F T    0.7*0.1*0.5/0.085=0.4118
F F    0.7*0.9*0.5/0.365=0.8630
"
  [point]
  (match [point]
         [[_ 1 1 _]] 0.0689
         [[_ 1 0 _]] 0.4030
         [[_ 0 1 _]] 0.4118
         [[_ 0 0 _]] 0.8630))

(defn pr-given-bap
"   
P(r|bap)=P(rap|b)*P(b)/P(bap)
        =(P(p|rab)*P(rab)/P(b))*P(b)/P(bap)
        =(P(p|rab)*P(rab))/P(bap)
       
P(bap)=P(b)*P(a|b)*(P(r|b)*P(p|ra)+P(~r|b)*P(p|~ra))
    b a p    P(bap)
    T T T    0.5*0.1*(0.3*0+0.7*0.6)=0.021
    T T F    0.5*0.1*(0.3*1+0.7*0.4)=0.029
    T F T    0.5*0.9*(0.3*0.8+0.7*1)=0.423
    T F F    0.5*0.9*(0.3*0.2+0.7*0)=0.027
    F T T    0.5*0.5*(0.8*0+0.2*0.6)=0.030
    F T F    0.5*0.5*(0.8*1+0.2*0.4)=0.220
    F F T    0.5*0.5*(0.8*0.8+0.2*1)=0.210
    F F F    0.5*0.5*(0.8*0.2+0.2*0)=0.040




P(r|bap) =(P(p|rab)*P(rab))/P(bap)
         =(P(p|ra)*P(r|b)*P(a|b)*P(b))/P(bap)

b a p    P(r)
T T T    0                     =0
T T F    1*0.3*0.1*0.5/0.029   =0.5172
T F T    0.8*0.3*0.9*0.5/0.423 =0.2553
T F F    0.2*0.3*0.9*0.5/0.027 =1
F T T    0                     =0
F T F    1*0.8*0.5*0.5/0.220   =0.9091
F F T    0.8*0.8*0.5*0.5/0.210 =0.7619
F F F    0.2*0.8*0.5*0.5/0.040 =1
"
  [point]
  (match [point]
         [[1 _ 1 1]] 0
         [[1 _ 1 0]] 0.5172
         [[1 _ 0 1]] 0.2553
         [[1 _ 0 0]] 1
         [[0 _ 1 1]] 0
         [[0 _ 1 0]] 0.9091
         [[0 _ 0 1]] 0.7619
         [[0 _ 0 0]] 1))
         
(defn pa-given-brp [point]
  [point]
  (match [point]
         [[1 1 _ 1]] 0
         [[1 1 _ 0]] 0.3571
         [[1 0 _ 1]] 0.0629
         [[1 0 _ 0]] 1
         [[0 1 _ 1]] 0
         [[0 1 _ 0]] 0.8333
         [[0 0 _ 1]] 0.375
         [[0 0 _ 0]] 1))

(defn pp-given-bra [point]
  (match [point]
         [[_ 1 1 _]] 0
         [[_ 1 0 _]] 0.8
         [[_ 0 1 _]] 0.6
         [[_ 0 0 _]] 1))

(def prob-position-table [[pb-given-rap 0]
                          [pr-given-bap 1]
                          [pa-given-brp 2]
                          [pp-given-bra 3]])

(defn gibbs-update [point n-iter]
  (let [prob-position-coll (partition 2 (flatten (repeat n-iter prob-position-table)))]
    (reduce (fn [point [prob position]]
              (if (< (draw (uniform-distribution)) (prob point))
                (assoc point position 1)
                (assoc point position 0)))
            point
            prob-position-coll)))

(defn gibbs [n-samples n-iter-each-sample]
  (let [point (vec (map #(if (<= % 0.5) 0 1) (stats/sample-uniform 4)))]
    (repeatedly n-samples
                #(gibbs-update point n-iter-each-sample))))

     
(defn build-query
  "[sample] -> (sample -> prob)"
  [samples]
  (fn [point]
    (/ (count (filter (partial = point)
                      samples))
       (count samples))))

(defn exam-panic-joint []
  (let [query (build-query (gibbs 10000 20))]
    (for [b [0 1] r [0 1] a [0 1] p [0 1]]
      [[b r a p] (query [b r a p])])))
              