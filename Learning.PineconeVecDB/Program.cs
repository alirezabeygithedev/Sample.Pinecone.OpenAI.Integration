using Azure.AI.OpenAI;
using Pinecone;

string pineconeApiKey = your_pinecone_api_key;
string openaiApiKey = your_openai_api_key;
string indexName = "main";
string pineconeEnv = "gcp-starter";

using var pinecone = new PineconeClient(pineconeApiKey, pineconeEnv);
await EnsureIndexExists(pinecone, indexName);
var openai = new OpenAIClient(openaiApiKey);
using var index = await pinecone.GetIndex(indexName);
var item1 = CreateVector("item1", await Embed("Change is the nature of life!", openai), new() { ["age"] = 14, ["length"] = 1023 });
var item2 = CreateVector("item2", await Embed("Discussion is the key!", openai), new() { ["age"] = 15, ["length"] = 1044 });
var item3 = CreateVector("item3", await Embed("Do not multiply yourself by zero!", openai), new() { ["age"] = 16, ["length"] = 1055 });
var item4 = CreateVector("item4", await Embed("Quality is good,Value is better!", openai), new() { ["age"] = 17, ["length"] = 1067 });
var item5 = CreateVector("item5", await Embed("Nothing lasts for ever!", openai), new() { ["age"] = 18, ["length"] = 1109 });
var item6 = CreateVector("item6", await Embed("What is this?", openai), new() { ["age"] = 19, ["length"] = 1011 });

await index.Upsert(new[] {
    item1,
    item2,
    item3,
    item4,
    item5,
    item6
});

var ageRange = new MetadataMap
{
    ["age"] = new MetadataMap
    {
        ["$gte"] = 16,
        ["$lte"] = 18
    }
};

var results = await index.Query(
    values: await Embed("Change is the nature of life!", openai),
    topK: 3,
    filter: ageRange,
    includeMetadata: true
    );

Console.WriteLine(string.Join('\n', results.SelectMany(v => v.Metadata!)));

await index.Delete(new[] {
    "item1",
    "item2",
    "item3",
    "item4",
    "item5",
    "item6",
});

Vector CreateVector(string id, float[] values, MetadataMap? metadata = null)
{
    return new Vector
    {
        Id = id,
        Values = values,
        Metadata = metadata
    };
}

async Task<float[]> Embed(string text, OpenAIClient openai)
{
    var request = new EmbeddingsOptions(text);
    var response = await openai.GetEmbeddingsAsync("text-embedding-ada-002", request);
    return response.Value.Data[0].Embedding.ToArray();
}

async Task EnsureIndexExists(PineconeClient pinecone, string indexName)
{
    if (!(await pinecone.ListIndexes()).Contains(indexName))
    {
        await pinecone.CreateIndex(indexName, 1536, Metric.Cosine);
    }
}