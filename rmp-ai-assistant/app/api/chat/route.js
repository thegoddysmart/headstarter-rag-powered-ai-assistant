import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import * as genai from 'google.generativeai'

// Step 2: Define the system prompt
const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`

// Step 3: Create the POST function
export async function POST(req) {
  const data = await req.json()

  // Step 4: Initialize Pinecone and Google Gemini
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  })
  const index = pc.index('rag').namespace('ns1')

  // Initialize Google Gemini
  genai.configure({ apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY })

  // Step 5: Process the user’s query
  const text = data[data.length - 1].content

  // Create an embedding using Google Gemini
  const embeddingResult = await genai.embedContent({
    model: 'models/text-embedding-004',  // Ensure this matches the correct embedding model for Google Gemini
    content: text,
    taskType: 'retrieval_document'
  })

  const embedding = embeddingResult.data[0].embedding

  // Step 6: Query Pinecone
  const results = await index.query({
    topK: 5,
    includeMetadata: true,
    vector: embedding,
  })

  // Step 7: Format the results
  let resultString = ''
  results.matches.forEach((match) => {
    resultString += `
    Returned Results:
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`
  })

  // Step 8: Prepare the Google Gemini request
  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

  // Step 9: Send request to Google Gemini
  const completion = await genai.generateText({
    model: 'gemini-1.5-flash',
    prompt: systemPrompt + '\n' + lastMessageContent,
    stream: true,
  })

  // Step 10: Set up streaming response
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      try {
        for await (const chunk of completion) {
          const content = chunk.text  // Adjusted to match the expected response structure from Google Gemini
          if (content) {
            const text = encoder.encode(content)
            controller.enqueue(text)
          }
        }
      } catch (err) {
        controller.error(err)
      } finally {
        controller.close()
      }
    },
  })

  return new NextResponse(stream)
}
