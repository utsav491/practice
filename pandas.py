import numpy as np
import pandas as pd


import pandas as pd

revenue = pd.DataFrame({
          "city":["Austin" , "Denver" , "Springfield" ,"Mendocino"],
           "branch_id" :[10,20,31,47] ,
            "revenue" :[100,83,4,200]  ,           
})



managers = pd.DataFrame({"city" :["Austin" ,"Denver" ,"Mendocino" ,"Springfield"],
                         "branch_id" :[10,20,47,30],
                         "manager":["Charles" , "Joel" ,"Brett", "Sally"]
                         })

    
print(revenue , managers)

merge_by_city = pd.merge(left  = revenue, right  = managers  ,on = "city")
print(merge_by_city)


# Merge revenue & managers on 'city' & 'branch': combined
combined = pd.merge(left = revenue , right = managers , left_on = "city" , right_on = "city")

# Print combined
print(combined)



# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX','CO','IL','CA']

# Add 'state' column to managers: managers['state']
managers['state'] = ['TX','CO','CA','MO']
print(revenue ,"\n", managers)

# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers , on =["branch_id" ,"city" , "state"] )

# Print combined
print(combined)
