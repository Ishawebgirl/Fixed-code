import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import Icon from './Icon'

interface IMenuItem {
  label: string
  icon: string
  linkTo?: string | null
  onClick?: () => void
  className?: string | null
  children?: React.ReactNode
  ref?: React.RefObject<HTMLDivElement>
}

const MenuItem = ({
  label,
  icon,
  linkTo,
  onClick,
  className,
  children,
  ref,
}: IMenuItem) => {
  useEffect(() => {}, [])

  return (
    <>
      {linkTo ? (
        <Link to={linkTo} relative="path" onClick={onClick} className={className ?? ''}>
          <Icon name={icon}></Icon> {label}
        </Link>
      ) : (
        <div onClick={onClick} className={className ?? ''} ref={ref}>
          <Icon name={icon}></Icon> {label}
        </div>
      )}
      {children}
    </>
  )
}

export default MenuItem
