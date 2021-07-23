library(gRchain)
library(jsonlite)

args <- commandArgs(TRUE)
# print(args[2])
# print(args[3])

df <- as.data.frame(fromJSON(txt=args[1]))
# print(names(df))
# print(head(df))

form <- as.formula(args[2])
# print(form)

types <- as.list(strsplit(args[3], ",")[[1]])
# print(types)

df <- data.frame(lapply(df, function(x) unlist(x)))

reggraph <- coxwer(form,
                    vartype=types, 
                    # data=sample_n(df, 100))
                    data=df)

# reggraph <- coxwer(#Oklevel~
#                     kumKKI_woM~ 
#                     KKI1_woM + Bej.sz.Matek ~ 
#                     #Treatment  ~ 
#                     Null.pont ~ 
#                     nem + Kor + Hoz.pont + Erett.pont + Tobb.pont,
#                     vartype=c(#"cont",
#                     "cont",
#                     "cont","count",
#                     #"bin",
#                     "cont",
#                     "bin","count","cont","cont", "cont"), 
#                     data=sample_n(df, 1000))

write.csv(reggraph$AdjMat, "reggraph_adjmat.csv", fileEncoding = "UTF-8")