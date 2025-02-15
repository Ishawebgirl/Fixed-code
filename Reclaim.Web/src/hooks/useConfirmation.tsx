import { confirmAlert } from 'react-confirm-alert'

const useConfirmation = () => {
  const confirm = (
    title: string,
    message: string,
    onConfirm: () => void,
    className?: string | null
  ) => {
    confirmAlert({
      customUI: ({ onClose }) => (
        <div className={`react-confirm-alert-body ${className}`}>
          <h1>{title}</h1>
          <div className="message" dangerouslySetInnerHTML={{ __html: message }} />
          <div className="react-confirm-alert-button-group">
            <button
              onClick={() => {
                onConfirm()
                onClose()
              }}
              className="ok">
              OK
            </button>
            <button
              onClick={() => {
                onClose()
              }}
              className="cancel">
              Cancel
            </button>
          </div>
        </div>
      ),
    })
  }

  return { confirm }
}

export default useConfirmation
