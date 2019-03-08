
import pandas as pd

revenue =pd.DataFrame({"city" :["Austin" ,"Denver" ,"Mendocino" ,"Springfield"],
                       "state" :["TX", "CO","IL","MO"],
                       "branch_id" :[10,20,30,47],
                       "revenue" :[100,83,4,200]
                       })

    
sales = pd.DataFrame({
                        "city" :["Austin" ,"Denver" ,"Mendocino" ,"Springfield","Springfield"],
                        "state" :[ "TX" , "CO" ,"IL" , "MO" , "IL" ] 
                        ,"units":[1,4,2,5,1]
                        })


    
    
managers = pd.DataFrame({"branch" :["Austin" ,"Denver" ,"Mendocino" ,"Springfield"],
                         "branch_id" :[10,20,47,30],
                         "manager":["Charles" , "Joel" ,"Brett", "Sally"] ,
                         "state" :["TX" ,"CO","IL", "MO"]
                         })

    
print(sales)

revenue_and_sales = pd.merge( left  = revenue , right = sales  ,on =["city" ,"state"]  ,how = "left")
print(revenue_and_sales)


revenue_and_sales = pd.merge( left  = revenue , right = sales  ,on =["city" ,"state"]  ,how = "right")
print(revenue_and_sales)


sales_and_managers = pd.merge(left =  sales , right = managers , right_on =["branch" , "state"] , left_on = ["city","state" ] )
print(sales_and_managers)


# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers , revenue_and_sales)

# Print merge_default
print(merge_default)

# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers , revenue_and_sales, how ="outer" )

# Print merge_outer
print(merge_outer)

# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers , revenue_and_sales , on = ["city" , "state"], how = "outer")

# Print merge_outer_on
print(merge_outer_on)
