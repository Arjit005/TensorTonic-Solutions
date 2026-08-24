import numpy as np
def detect_drift(reference_counts: list, production_counts: list, threshold: float) -> dict:
    """Return the TVD score and whether it exceeds the threshold."""
    # Write code here
    # convert input into np array
    reference_counts=np.asarray(reference_counts,dtype=float)
    production_counts = np.asarray(production_counts, dtype=float)

    """
    The problem explicitly says:

        "Divide every bin count by the total count of its histogram."
        
        So your brain should translate that into:
        
        total = sum(counts)
        normalized = counts / total
    """
    
    #Normalize both histograms independently
    reference_total = np.sum(reference_counts)
    production_total = np.sum(production_counts)

    norm_reference_counts = reference_counts / reference_total
    norm_production_counts = production_counts / production_total

    
    absolute_diffrence=abs(norm_reference_counts-norm_production_counts)
    total_distance=np.sum(absolute_diffrence)
    drift=total_distance/2

    #Drift is detected when score is strictly greater than threshold.
    res={"score":drift}
    if drift>threshold:
        res["drift_detected"]=True
        return res
    else:
        res["drift_detected"]=False
        return res