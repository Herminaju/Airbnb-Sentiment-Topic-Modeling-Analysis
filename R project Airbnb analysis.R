#Analysis with Unstructured data
#Student Name: Hermina Judith Chinnappan


#----------------------------------
# Step 1: Setting up the Environment
#----------------------------------

# Install necessary packages

#install.packages("mongolite")
#install.packages("dplyr")
#install.packages("tidytext")
#install.packages("ggplot2")
#install.packages("tidyr")
#install.packages("topicmodels")
#install.packages("tm")
#install.packages("factoextra")
#install.packages("textdata")
#install.packages("igraph")
#install.packages("ggraph")
#install.packages("widyr")
#install.packages("stringr")
#install.packages("purrr")
#install.packages("textcat")

# Load necessary packages
library(mongolite)
library(dplyr)
library(tidytext)
library(ggplot2)
library(tidyr)
library(topicmodels)
library(tm)
library(factoextra)
library(textdata)
library(igraph)
library(ggraph)
library(widyr)
library(stringr)
library(purrr)
library(textcat)

#----------------------------------
# Step 2: Connecting to MongoDB and Loading Data
#----------------------------------

connection_string <- 'mongodb+srv://hermina:herminaisapirate@mongo-learning.hib9j.mongodb.net/'
airbnb_collection <- mongo(collection="listingsAndReviews", db="sample_airbnb", url=connection_string)
airbnb_all <- airbnb_collection$find()

#----------------------------------
# Step 3: Data Preparation
#----------------------------------

text_data <- airbnb_all %>%
  mutate(lang = textcat(description)) %>%
  mutate(text_content = description)  %>%
  filter(lang == "english") 

# Tokenize and clean text data
tidy_text <- text_data %>%
  unnest_tokens(word, text_content) %>%
  anti_join(stop_words) 

#----------------------------------
# Text Mining 
#----------------------------------
#----------------------------------
# Step 4: Sentiment Analysis (BING, NRC, AFINN)
#----------------------------------
sentiments_bing <- get_sentiments("bing")
sentiments_nrc <- get_sentiments("nrc")
sentiments_afinn <- get_sentiments("afinn")

bing_analysis <- tidy_text %>%
  inner_join(sentiments_bing, by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  mutate(method = "BING")

nrc_analysis <- tidy_text %>%
  inner_join(sentiments_nrc, by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  mutate(method = "NRC")

afinn_analysis <- tidy_text %>%
  inner_join(sentiments_afinn, by = "word") %>%
  mutate(sentiment = ifelse(value > 0, "positive", "negative")) %>%
  count(sentiment, sort = TRUE) %>%
  mutate(method = "AFINN")

# Merging the sentiment data
total_sentiment <- bind_rows(bing_analysis, nrc_analysis, afinn_analysis)

# Visualization
combined_plot <- ggplot(total_sentiment, aes(x = sentiment, y = n, fill = method)) +
  geom_col(position = "dodge", show.legend = TRUE) +
  labs(title = "Sentiment Analysis (BING, NRC, AFINN)",
       x = "Sentiment", y = "Count") +
  facet_wrap(~method, scales = "free_y") +
theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(combined_plot)

##############################################################
######## Most Common Positive and Negative Words #############
##############################################################

# Count positive and negative words using Bing lexicon
bing_counts <- tidy_text %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

# Display the most common positive and negative words
bing_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment", x = NULL, title = "Most Common Positive and Negative Words") +
  coord_flip()

#----------------------------------
# Step 5: Topic Modeling with LDA
#----------------------------------

tdm <- tidy_text %>%
  count(document = row_number(), word) %>%
  cast_dtm(document, word, n)

lda_model <- LDA(tdm, k = 5, control = list(seed = 123))
topics <- tidy(lda_model, matrix = "beta")

# Extracting top terms per topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

print(top_terms)

# Visualization of top terms
ggplot(top_terms, aes(x = reorder(term, beta), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Top Terms per Topic in LDA", x = "Term", y = "Beta")

#----------------------------------
# Step 6: N-grams and Word Frequency Analysis
#----------------------------------

bigrams <- text_data %>%
  unnest_tokens(bigram, description, token = "ngrams", n = 2)

#Remove stopwords in bigram
bigrams_separated <- bigrams %>%
  separate(bigram, into = c("word1", "word2"), sep = " ")

stop_words_list <- get_stopwords()

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words_list$word,
         !word2 %in% stop_words_list$word)

#Recombining filtered words
bigrams_clean <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

#Counting frequency
bigram_counts <- bigrams_clean %>%
  count(bigram, sort = TRUE)

# Select the 20 most frequent bigrams
top_bigrams <- bigram_counts %>%
  top_n(20, n)

#Vizualisation
ggplot(top_bigrams, aes(x = reorder(bigram, n), y = n)) +
  geom_col(fill = "#2c3e50") +
  coord_flip() +
  labs(title = "Common Bigrams in Airbnb Descriptions",
       x = "Bigram",
       y = "Frequency") +
  theme_minimal()

#######################################################
### Word Frequency Correlation: Apartment vs Others ###
#######################################################

# Filter data based on property types
Apartment <- airbnb_all %>%
  filter(property_type == 'Apartment')

Condominium <- airbnb_all %>%
  filter(property_type == 'Condominium')

Serviced_apartment <- airbnb_all %>%
  filter(property_type == 'Serviced apartment')

# Tokenize and Clean Data
tidy_apartment <- Apartment %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words)

tidy_condominium <- Condominium %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words)

tidy_serviced <- Serviced_apartment %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words)

# Combining datasets and calculating frequencies
frequency <- bind_rows(mutate(tidy_apartment, type="Apartment"),
                       mutate(tidy_condominium, type="Condominium"),
                       mutate(tidy_serviced, type="Serviced")) %>%
  mutate(word = str_extract(word, "[a-z']+")) %>%
  count(type, word) %>%
  group_by(type) %>%
  mutate(proportion = n / sum(n)) %>%
  select(-n) %>%
  spread(type, proportion) %>%
  gather(type, proportion, `Condominium`, `Serviced`)

# Plotting the correlation
ggplot(frequency, aes(x = proportion, y = `Apartment`, color = abs(`Apartment` - proportion))) +
  geom_abline(color = "grey40", lty = 2) +
  geom_jitter(alpha = .1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = scales::percent_format()) +
  scale_y_log10(labels = scales::percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  facet_wrap(~type, ncol = 2) +
  theme(legend.position = "none") +
  labs(y = "Apartment", x = NULL, 
       title = "Word Frequency Correlation: Apartment vs Condominium & Serviced Apartments")

############################################################
### Step 7: Availability and Hotel Count Analysis by Country
############################################################

# Extract country from nested address and calculate availability
country_availability <- airbnb_all %>%
  mutate(country = address$country, 
         availability_count = availability$availability_365) %>%
  group_by(country) %>%
  summarise(total_availability = sum(availability_count, na.rm = TRUE),
            hotel_count = n()) %>%
  arrange(total_availability)

# Display the availability count and hotel count by country
print(country_availability)

# Visualization: Countries with Least Availability
ggplot(country_availability, aes(x = reorder(country, total_availability), y = total_availability)) +
  geom_col(fill = "coral") +
  coord_flip() +
  labs(title = "Availability by Country",
       x = "Country", y = "Total Availability (365 Days)")

# Visualization: Number of Hotels by Country
ggplot(country_availability, aes(x = reorder(country, hotel_count), y = hotel_count)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(title = "Number of Hotels by Country",
       x = "Country", y = "Total No of Hotels")

#############################################
### Step 8: Average Price by State (US Only)
#############################################

# Filter for US listings and calculate average price by state
us_price <- airbnb_all %>%
  filter(address$country_code == "US", !is.na(price) & price > 0) %>%
  group_by(state = address$market) %>%
  summarise(average_price = mean(price, na.rm = TRUE)) %>%
  arrange(desc(average_price))

# Display the average price by state
print(us_price)

# Visualization: Average Price by State
ggplot(us_price, aes(x = reorder(state, average_price), y = average_price)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Average Price by State (US)",
       x = "State", y = "Average Price in Dollar's")

