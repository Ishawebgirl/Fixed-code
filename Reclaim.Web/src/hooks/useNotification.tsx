import { useState } from 'react'
import { confirmAlert } from 'react-confirm-alert'

const useNotification = () => {
  const [nonce, setNonce] = useState(1)

  const notify = (title: string, message: string, className?: string | null) => {
    confirmAlert({
      customUI: ({ onClose }) => (
        <div className={`react-confirm-alert-body ${className}`}>
          <h1>{title}</h1>
          <div className="message" dangerouslySetInnerHTML={{ __html: message }} />
          <div className="react-confirm-alert-button-group">
            <button
              onClick={() => {
                setNonce(nonce + 1)
                onClose()
              }}
              className="ok">
              OK
            </button>
          </div>
        </div>
      ),
    })
  }

  return { notify, nonce }
}

export default useNotification
