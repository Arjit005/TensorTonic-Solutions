def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    # Write code here
    # R=unique recommended items
    #N=N n_items

    # type hinting se muje ek chij samajh aa rhi hai ki muje float may output chaiye
    unique_items=[]
    n=len(recommendations)
   
    if not recommendations:
         return 0.0
    for i in range(n):
        for j in range(len(recommendations[i])):
            if recommendations[i][j] not in unique_items:
                unique_items.append(recommendations[i][j])
            else:
                pass
    
    total_unique_items=len(unique_items)
    if n_items==0:
        return 0.0
    
    catalog=total_unique_items/n_items
    return catalog






    
  