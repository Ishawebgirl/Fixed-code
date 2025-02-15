import * as Model from 'api/model'
import { useParams, useNavigate } from 'react-router-dom'
import { useMemo, useState, useEffect, useRef } from 'react'
import { Log } from 'helpers/log'
import Menu from 'components/Menu'
import MenuItem from 'components/MenuItem'
import Icon from 'components/Icon'
import ReactMarkdown from 'react-markdown'
import DocumentViewer from 'components/DocumentViewer'

const ClaimChat = () => {
  const { uniqueID, chatUniqueID } = useParams()
  const navigate = useRef(useNavigate())

  const claimApi = useMemo(() => new Model.ClaimClient(process.env.REACT_APP_API_URL), [])

  const [chat, setChat] = useState<Model.Chat>()
  const messagesEndRef = useRef<null | HTMLDivElement>(null)
  const inputBoxOuterRef = useRef<null | HTMLDivElement>(null)
  const [question, setQuestion] = useState('')
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [isStarted, setIsStarted] = useState<boolean | null>(null)
  const [clickedDocument, setClickedDocument] = useState<Model.Document>()
  const [citations, setCitations] = useState<Model.Citation[] | undefined>()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    const initialInputBottom = window.innerHeight / 2 - 100

    if (isStarted !== null) {
      inputBoxOuterRef.current?.style.setProperty(
        'bottom',
        (isStarted ? 0 : initialInputBottom) + 'px'
      )
    }
  }, [isStarted])

  useEffect(() => {
    setQuestion('')
    setSubmittedQuestion('')
    setIsThinking(false)

    if (chat && chat.messages) {
      setIsStarted(chat.messages.length > 0)
    }

    scrollToBottom()
  }, [chat])

  useEffect(() => {
    scrollToBottom()
  }, [isThinking])

  const submitQuestion = () => {
    setIsStarted(true)
    setIsThinking(true)
    setSubmittedQuestion(question)
    setQuestion('')
    onQuestionSubmit(question)
  }

  useEffect(() => {
    if (!claimApi || !uniqueID) {
      return
    }

    const fetchData = async () => {
      try {
        if (chatUniqueID === undefined) {
          const result = await claimApi.createChat(uniqueID)
          setChat(result)
          navigate.current(result.uniqueID)
        } else if (chatUniqueID) {
          const result = await claimApi.getChat(uniqueID, chatUniqueID)
          setChat(result)
          scrollToBottom()
        }
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    }

    fetchData()
  }, [claimApi, uniqueID, chatUniqueID])

  const onQuestionSubmit = async (question: string) => {
    if (question.trim() === '') {
      return
    }

    if (chat) {
      const newChat = await claimApi.queryClaim(uniqueID!, chatUniqueID!, question)
      setChat(newChat)
    }
  }

  const onCitationClick = async (
    clickedChatMessage: Model.ChatMessage,
    clickedDocumentUniqueID: string
  ) => {
    const relevantCitations = clickedChatMessage.citations.filter(
      (x) => x.documentUniqueID === clickedDocumentUniqueID
    )
    setCitations(relevantCitations)

    const dummyDocument = new Model.Document()
    dummyDocument.uniqueID = clickedDocumentUniqueID
    dummyDocument.name = relevantCitations[0].fileName

    setClickedDocument(dummyDocument)
  }

  return (
    chat && (
      <>
        <div className="header">Chat with claim {chat.claimExternalID}</div>
        <Menu>
          <MenuItem
            label="Return to claim"
            icon="ArrowLeft"
            linkTo={`/claims/${uniqueID}`}
          />
        </Menu>
        <div id="chat">
          <div className="messages">
            {chat.messages.map((message: Model.ChatMessage, index: number) => (
              <div>
                <div
                  key={index}
                  className={`message ${message.chatRole.toLowerCase()}`}
                  ref={
                    !isThinking && index === chat.messages.length - 1
                      ? messagesEndRef
                      : null
                  }>
                  <ReactMarkdown>{message.text}</ReactMarkdown>
                </div>
                <div className="citations">
                  {Array.from(
                    new Set(
                      message.citations.map((citation) => citation.documentUniqueID)
                    )
                  ).map((documentUniqueID, index) => {
                    const citation = message.citations.find(
                      (c) => c.documentUniqueID === documentUniqueID
                    )
                    return (
                      <span
                        key={index}
                        className="citation"
                        onClick={() => onCitationClick(message, documentUniqueID)}>
                        {citation!.fileName}
                      </span>
                    )
                  })}
                </div>
              </div>
            ))}
            {isThinking && (
              <>
                <div className="message user">{submittedQuestion}</div>
                <div className="message thinking" ref={messagesEndRef}>
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </>
            )}
          </div>
          {isStarted !== null ? (
            <div
              className={`input-box-outer ${isStarted === false ? 'initial' : ''}`}
              ref={inputBoxOuterRef}>
              <div className="input-box">
                <span className="instructions">
                  Use the textbox below to ask a question about the claim. The system will
                  interrogate all associated documents for the claim.
                </span>
                <textarea
                  rows={3}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      submitQuestion()
                    }
                  }}
                  onChange={(e) => setQuestion(e.target.value)}
                  value={question}
                />
              </div>
              <Icon
                onClick={submitQuestion}
                className="send-button"
                name="PaperPlane"></Icon>
            </div>
          ) : null}
        </div>
        <DocumentViewer
          theDocument={clickedDocument}
          claimUniqueID={uniqueID}
          citations={citations}
          onCancel={() => {
            setClickedDocument(undefined)
          }}></DocumentViewer>
      </>
    )
  )
}

export default ClaimChat
