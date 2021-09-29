fpath = "/home/ms314/projects/tagnav-code-refactored/data/processed/data_survey_books.csv"
t<-read.table(fpath, header=TRUE, sep=",", quote = '"')
write.table(t, file=fpath, sep=",", col.names = TRUE, row.names = FALSE, fileEncoding = "utf-8", quote=FALSE)
