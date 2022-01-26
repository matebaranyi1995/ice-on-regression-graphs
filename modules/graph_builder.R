library(gRchain)
library(jsonlite)

args <- commandArgs(TRUE)
# print(args[2])
# print(args[3])

df <- as.data.frame(fromJSON(txt=args[1]))
# print(names(df))
# print(head(df))
print(str(df))
print(sapply(df, class))

form <- as.formula(args[2])
# print(form)

# types <- as.list(strsplit(args[3], ",")[[1]])
# print(types)


col_names <- sapply(df, function(col) length(unique(col)) < 5)
df[, col_names] <- sapply(df[, col_names], as.factor)

df[sapply(df, is.character)] <- sapply(df[sapply(df, is.character)], as.factor)

df <- data.frame(lapply(df, function(x) unlist(x)))

print(str(df))
print(sapply(df, class))
reggraph <- coxwer(form,
                    # vartype = types,
                    # data=sample_n(df, 100))
                    data = df)

# # calling example
# reggraph <- coxwer(Oklevel~
#                     kumKKI_woM~
#                     KKI1_woM + Bej.sz.Matek ~
#                     Treatment  ~
#                     Null.pont ~
#                     nem + Kor + Hoz.pont + Erett.pont + Tobb.pont,
#                     vartype=c("cont",
#                     "cont",
#                     "cont","count",
#                     "bin",
#                     "cont",
#                     "bin","count","cont","cont", "cont"),
#                     data=sample_n(df, 1000))

write.csv(reggraph$AdjMat, "tmp/reggraph_adjmat.csv", fileEncoding = "UTF-8")