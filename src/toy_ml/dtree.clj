(ns toy-ml.dtree
    (:use [incanter core io]))

(defn entropy [distribution]
  (sum (map #(if (= 0 %) 0
                 (- ($= % * (log2 %))))
            distribution)))

(defn gini [distribution]
  (- 1 (sum (map #($= % ** 2)
                 distribution))))

(defn- seq-made-sure [x]
  (if (sequential? x) x
      [x]))

(defn data-gain [ds target gain-fn]
  (let [target-count-coll
        (seq-made-sure ($ :target-count
                          ($rollup :count :target-count
                                   target ds)))
        s (sum target-count-coll)
        prob-distribution ($= target-count-coll / s)]
    (gain-fn prob-distribution)))
    
(defn- ds-size [ds] (first (dim ds)))

(defn info-gain [ds target feature gain-fn]
  (let [entire (data-gain ds target gain-fn)
        grouped-map ($group-by feature ds)]
    ($= entire - 
        (sum (map (fn [sub-ds] (* (data-gain sub-ds target gain-fn)
                                  ($= (ds-size sub-ds) / (ds-size ds))))
                  (vals grouped-map))))))


(defn- most-common [coll]
  (->> (group-by identity coll)
      vals
      (sort-by count)
      last first))

(declare dtree make-node)

(defn dtree [ds target &
             {:keys [gain-fn] :or {gain-fn entropy}}]
  (cond (= 1 (count ($group-by target ds)))
        (if (sequential? ($ target ds))
          {:value (first ($ target ds))}
          {:value ($ target ds)})
        
        (= 1 (second (dim ds)))
        {:value (most-common ($ target ds))}
        
        true
        (make-node ds target gain-fn)))

(defn- choose-feature [ds target gain-fn]
  (let [col-names (remove #(= % target) (col-names ds))
        gains (map #(info-gain ds target % gain-fn) col-names)
        gain-col-map (zipmap gains col-names)]
    (-> gain-col-map sort last val)))

(defn- make-children [ds feature target gain-fn]
  (map (fn [[key sub-ds]]
         (if (= 1 (nrow sub-ds))
           [key (dtree sub-ds target :gain-fn gain-fn)]
           [key (dtree ($ [:not feature] sub-ds)
                        target :gain-fn gain-fn)]))
       ($group-by feature ds)))

(defn- make-node [ds target gain-fn]
  (let [feature (choose-feature ds target gain-fn)
        children (make-children ds feature target gain-fn)]
    {:node feature :children children}))