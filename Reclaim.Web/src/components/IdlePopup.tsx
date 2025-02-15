import { useEffect, useRef, useState } from 'react'
import configSettings from 'settings/config.json'

interface IIdlePopup {
  isOpen?: boolean
  onClose: (isLogout: boolean) => void
  remainingTime?: number
}

const IdlePopup = ({ isOpen, onClose, remainingTime }: IIdlePopup) => {
  const [time, setTime] = useState(Number(configSettings.IdlePopupDuration))
  const mounted = useRef(0)

  useEffect(() => {
    if (isOpen) {
      if (time <= 1) {
        if (mounted.current) {
          window.clearTimeout(mounted.current)
        }
        onClose(true)
        return
      }
      mounted.current = window.setTimeout(() => setTime(time - 1), 1000)
    } else {
      setTime(Number(configSettings.IdlePopupDuration))
    }
    return () => {
      if (mounted.current) {
        window.clearTimeout(mounted.current)
      }
    }
  }, [isOpen, onClose, time])

  return (
    <div className={`top-modal${isOpen ? ' is-open' : ''}`}>
      <div className="top-modal-frame">
        <div className="top-modal-content">
          You&apos;ve been idle for a while. You&apos;ll be logged out in {remainingTime}{' '}
          seconds, unless you&apos;re still there!
        </div>
        <div className="top-modal-buttons">
          <button className="button" onClick={() => onClose(false)}>
            I&apos;m still here!
          </button>
          <button className="button is-danger" onClick={() => onClose(true)}>
            Logout
          </button>
        </div>
      </div>
    </div>
  )
}

export default IdlePopup
