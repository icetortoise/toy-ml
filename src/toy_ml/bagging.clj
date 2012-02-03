(ns toy-ml.bagging
  (:use [toy-ml dtree core]
        [incanter core distributions]))

(defn sample [ds]
  (let [rows-chosen
        (repeatedly (nrow ds) #(draw (range (nrow ds))))]
    ($ rows-chosen :all ds)))

(defn bagging-classifiers [samples-seq dtree target]
  (map (fn [ds] (dtree ds target)) samples-seq))

(defn bagging-classify [ds trees]
  (let [votes (vec (map #(dtree-classify-dataset % ds)
                        trees))
        votes-ds (to-dataset votes)]
    (for [c (col-names votes-ds)]
      (most-common ($ c votes-ds)))))
    
    
             