import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import Icon from './Icon'

interface INavItem {
  label: string
  icon: string
  iconStyle?: string
  href?: string
}

const NavItem = ({ label, icon, href = '' }: INavItem) => {
  const [isActive, setIsActive] = useState(false)

  const { pathname } = useLocation()

  useEffect(() => {
    setIsActive(location.pathname === href)
  }, [href, pathname])

  return (
    <li className="nav-item">
      <Link
        className={`nav-item-link${isActive ? ' active' : ''}${href === '' ? ' disabled-links' : ''}`}
        to={href}>
        <Icon name={icon} />
        <span className="nav-item-label">{label}</span>
      </Link>
    </li>
  )
}

export default NavItem
