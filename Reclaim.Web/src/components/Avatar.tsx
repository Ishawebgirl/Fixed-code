import { useEffect, useState } from 'react'
import Icon from './Icon'

interface IAvatar {
  id?: string
  url: string
  name: string
  className?: string
  onClick?: (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => void
}

const Avatar = ({ id, url, name, className, onClick }: IAvatar) => {
  const [initials, setInitials] = useState('??')

  useEffect(() => {
    if (name !== undefined && name.length > 0) {
      const parts = name.toUpperCase().split(' ')

      if (parts.length > 1) {
        setInitials(parts[0][0] + parts[1][0])
      } else if (parts.length === 1) {
        setInitials(parts[0][0] + parts[0][1])
      } else {
        setInitials(parts[0][0])
      }
    }
  }, [name])

  return (
    <div id={id} className={`avatar ${className}`} onClick={onClick}>
      {url.length === 0 ? (
        name.length === 0 ? (
          <div className="name">
            <Icon name="User"></Icon>
          </div>
        ) : (
          <div className="initials">{initials}</div>
        )
      ) : (
        <div className="image" style={{ backgroundImage: `url(${url})` }}></div>
      )}
    </div>
  )
}

export default Avatar
