"use client";

import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext'; // <-- Use the hook
import { Button } from './ui/button';
import { toast } from 'sonner';

export default function LogoutButton() {
  const { signOut } = useAuth();
  const router = useRouter();

  const handleLogout = async () => {
    await signOut();
    toast.success("You have been logged out.");
    router.push('/'); // Navigate to home page
  };

  return (
    <Button variant="destructive" onClick={handleLogout}>
      Logout
    </Button>
  );
}